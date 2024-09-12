import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import shapes
# Packages for Random Forest Segmentation:
from skimage import feature, future
from functools import partial
from glob import glob
import argparse
# packages for mask postprocessing:
import skimage.morphology as sk_morphology
import scipy.ndimage as sc_img
from FileHandler import *

np.random.seed(44)

parser = argparse.ArgumentParser(description='load data')
parser.add_argument('path', type=str,  # default='/media/mauro-gwerder/Elements_2TB/PHD/COMET/B08/',
                    help='path to image folder. Needs to contain a channel list and RFseg model in folder as well')
parser.add_argument('-c', '--channel', type=str, default='PanCK', help='Channel name to use for segmentation task')
parser.add_argument('-f', '--figures', action='store_false', help='If quality control figures should be saved')
parser.add_argument('-r', '--run', type=str, default='', help='Tag for your current run')

def main():
    global args
    args = parser.parse_args()

    # Handle argparse arguments
    PATH = args.path
    run_tag = args.run
    print('\n\n\nCurrent task: Epithelium segmentation\n\n\n')
    file_handler(PATH, 'EpiSegmentor')

    channel_name = args.channel
    figures = args.figures

    if run_tag != '':
        run_tag = '_' + run_tag

    tiles_path = PATH + 'tiles' + run_tag + '/'
    mask_path = tiles_path + 'masks_RFseg/'
    os.mkdir(mask_path) if not osp.exists(mask_path) else None
    QC_path = PATH + 'QC/'
    os.mkdir(QC_path) if not osp.exists(QC_path) else None

    model_path = glob(PATH + '*_model.pkl')[0]
    channel_path = glob(PATH + '*_channels.pkl')[0]

    if figures:
        figures_path = f'{tiles_path}plots/'
        os.mkdir(figures_path) if not osp.exists(figures_path) else None

    # Load RFseg model; train your own with the supplied jupyter notebook
    clf = pickle.load(open(model_path, 'rb'))

    tile_names = np.sort(glob(tiles_path + '/*.tiff'))
    #  Load channel names; for now no way to extract channel names from ome.tiff directly
    channel_dict = pickle.load(open(channel_path, 'rb'))
    dict_keys = list(channel_dict.keys())
    channel_id = np.where(channel_dict[dict_keys[0]] == channel_name)[0]
    # # Training of the classifier:
    # annots = gpd.read_file(ANNOT_PATH)
    # train_img = imread(TRAINING_IMAGE)
    # annot_dict = {'Tumor Bud': 1,
    #               'primary': 1,
    #               'background': 2,
    #               'Artifact': 2}
    # geom_id = ((geom, annot_dict[ID['name']]) for geom, ID in zip(annots['geometry'], annots['classification']))
    #
    # raster_annot = rasterio.features.rasterize(geom_id,
    #                                            out_shape=(train_img.shape[0], train_img.shape[1]),  # 3640,3643
    #                                            # transform = raster.transform,
    #                                            all_touched=True,
    #                                            fill=0,  # background value
    #                                            # merge_alg = MergeAlg.replace,
    #                                            dtype=np.int16)
    sigma_min = 1
    sigma_max = 6
    # feature extraction function; needs to match the features which were used to train the model
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            num_sigma=3,
                            channel_axis=-1)
    # train_img_rgb = channel_handling(train_img, PanCK_Channel)
    # features = features_func(train_img_rgb)
    # clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,  # criterion='log_loss',
    #                              max_depth=10, max_samples=0.05)
    # clf = future.fit_segmenter(raster_annot, features, clf)
    i = 0
    for tile in tile_names:
        basename, _ = osp.splitext(osp.splitext(tile)[0])
        path, file_basename = osp.split(basename)
        if osp.exists(f'{mask_path}{file_basename}_mask.ome.tiff'):
            continue
        print(f'current tile: {file_basename}')
        try:
            img = np.transpose(imread(tile), (2, 1, 0))
        except RuntimeError:
            print(f'Tile {file_basename} is corrupted. Please Re-run tile extraction for said tile.')
            if osp.exists(QC_path + 'corrupted_tiles.txt'):
                write_mode = 'a'
            else:
                write_mode = 'w'
            with open(QC_path + 'corrupted_tiles.txt', write_mode) as f:
                f.write(f'{file_basename}')
            continue
        # Convert image to rgb
        img_rgb = channel_handling(img, channel_id)
        features = features_func(img_rgb)
        result = future.predict_segmenter(features, clf)
        if figures:
            plt.subplot(1, 3, 1)
            plt.axis('off')
            plt.title('Original image')
            img_plt = img_rgb / (2**16)
            plt.imshow(img_plt)

        # Mask refinement
        mask = np.abs(result - 1)
        mask_refined = refine_mask(mask)
        if figures:
            plt.subplot(1, 3, 2)
            plt.title('Mask overlay')
            plt.axis('off')
            plt.imshow(img_plt)
            plt.imshow(mask_refined, alpha=0.5)

        test, mask_regions = cv2.connectedComponents(mask_refined.astype(np.uint8))
        if figures:
            plt.subplot(1, 3, 3)
            plt.title('Unique regions')
            plt.axis('off')
            plt.imshow(mask_regions)
            plt.savefig(f"{figures_path}{file_basename}_plot.png", dpi=300)

        imwrite(mask_path + file_basename + '_mask.ome.tiff', mask_regions)
        # polygons = mask_to_polygon(mask_refined)
        # combine polygons with encapsuled polygons to create holes in the segmentation.
        # polygons_with_holes = create_inner_polygons(polygons)
        # polygons_with_holes.to_file(basename + '_seg.geojson', driver='GeoJSON')
        i += 1
        print(f'Processed tiles: {i} out of {len(tile_names)}.')
    return None


def channel_handling(img, channel):
    """
    Reshuffle the channels to create an RGB image where:
    R = None
    G = Segmentation marker of your choice
    B = DAPI marker
    :param img: Image tile (DAPI has to be channel 0)
    :param channel: Index of segmentation marker of your choice
    :return: 3-Channel image ready for segmentation
    """
    img_rgb = np.zeros((np.shape(img)[0], np.shape(img)[1], 3), dtype=np.uint16)
    img_rgb[:, :, 1] = np.squeeze(img[:, :, channel]).astype(np.uint16)
    img_rgb[:, :, 2] = np.squeeze(img[:, :, 0]).astype(np.uint16)
    return img_rgb


def filter_remove_small_holes(np_img, min_size=300, output_type="uint8"):
    """
    Filter image to remove small holes less than a particular size.
    Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).
    Returns:
    NumPy array (bool, float, or uint8).
    """
    img_bool = np_img.astype(bool)  # make sure mask is boolean
    rem_sh = sk_morphology.remove_small_holes(img_bool, area_threshold=min_size)

    if output_type == "bool":
        pass
    elif output_type == "float":
        rem_sh = rem_sh.astype(float)
    else:
        rem_sh = rem_sh.astype("uint8") * 255
    return rem_sh


def filter_binary_erosion(np_img, disk_size=7, iterations=1, output_type="bool"):
    """
    Erode a binary object (bool, float, or uint8).
    Args:
        np_img: Binary image as a NumPy array.
        disk_size: Radius of the disk structuring element used for erosion.
        iterations: How many times to repeat the erosion.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where edges have been eroded.
    """

    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_img.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_remove_small_objects(np_img,
                                min_size=300,
                                output_type="uint8"):
    """
    Filter image to remove small objects (connected components) less than a particular
    minimum size. If avoid_overmask is True, this function can recursively call itself
    with progressively smaller minimum size objects to remove to reduce the amount of
    masking that this filter performs.
    Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    percentage value.
    output_type: Type of array to return (bool, float, or uint8).
    Returns:
    NumPy array (bool, float, or uint8).
    """

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def refine_mask(mask, max_hole_size=2000, disk_size=2, min_obj_size=900):
    """
    Applies 3 operations to the mask:
    -> Small objects are removed, resulting in a smoother mask
    -> Holes inside the binary are filled, resulting in a smoother mask
    -> Erosion leads to a more compact mask, with a lot of the debris being discarded
    :param mask: binary mask in form of an np.array.
    :param max_hole_size: maximum sizes of holes that are removed;
    :param disk_size: size of disk used for erosion;
    :param min_obj_size: minimum size of objects to not be removed;
    :return: Mask
    """

    holes_filled = filter_remove_small_holes(mask, min_size=max_hole_size)
    erosion_filtered = filter_binary_erosion(holes_filled, disk_size)
    small_removed = filter_remove_small_objects(erosion_filtered, min_size=min_obj_size)
    return small_removed


def mask_to_polygon(mask):
    """
    Converts numpy array mask to shapely polygons.
    mask: binary numpy array;
    Output: geopandas.GeoSeries of shapely.Polygons;
    """
    mask = mask.astype('uint8')
    shape = shapes(mask)
    shapes_from_mask = np.asarray([list(s) for s in shape])  # GeoSeries([s for s in shape])
    mask_shape_series = gpd.GeoSeries([Polygon(s['coordinates'][0]) for s in shapes_from_mask[:, 0]])
    num_polygons = len(mask_shape_series)
    # The last generated polygon is always a polygon of tile-size. We therefore
    # remove this polygon. This could, however, be used for a stroma mask.
    mask_shape_series = mask_shape_series[0:num_polygons - 1].buffer(3)

    return mask_shape_series


def create_inner_polygons(pol_series):
    """
    Checks for polygons within a list that are contained by another polygon in the same list.
    If that's the case, inner polygons and polygons will be combinedto polygons with holes.
    pol_series: geopandas.GeoSeries of shapely.polygons
    output: geopandas.GeoSeries of shapely.polygons
    """
    row_list = []
    for row, pol in enumerate(pol_series):
        # Drop identity polygons, so the query polygon is not compared to itself.
        df_dropped = pol_series.drop([pol_series.index[row]])
        df_holes = df_dropped[df_dropped.contains(pol)]
        if len(df_holes) > 0:
            ref_pol = df_holes[df_holes.index[0]]
            # Create polygons with holes
            pol_series[df_holes.index[0]] = ref_pol.difference(pol)
            row_list.append(row)
    # Remove inner polygons from list
    pol_series = pol_series.drop(row_list)
    return pol_series


if __name__ == '__main__':
    main()
