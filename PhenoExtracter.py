# ImportError: Numba needs NumPy 1.21 or less
# pip install numpy==1.21.6
import numpy as np
import xtiff
# Compatibility issue with cv2: whenever I want to plot something, cv2 complains about xcblib.
# Way to solve this issue: https://github.com/opencv/opencv-python/issues/386
import cv2
import geopandas
from rasterio import features
from glob import glob
from tifffile import imread
import pickle
from stardist.models import StarDist2D
from geopandas import GeoSeries
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
import pandas as pd
from csbdeep.utils import normalize
import os
import os.path as osp
import argparse
from FileHandler import *


np.random.seed(44)

parser = argparse.ArgumentParser(description='segment nuclei and extract expression patterns')
parser.add_argument('path', type=str, help='path to image folder')
parser.add_argument('-m', '--model', type=str, default='/home/mauro-gwerder/PycharmProjects/PhD_new/stardist_Lunaphore',
                    help='Path to StarDist model folder')
parser.add_argument('-s', '--save', type=str, default='', help='path to output folder')
parser.add_argument('-r', '--resume', type=str, default='',
                    help='path to existing output file which should be extended')
parser.add_argument('-d', '--dapi', type=int, default=0, help='ID of the DAPI channel. Default is 0')
parser.add_argument('-r', '--run', type=str, default='', help='Tag for your current run')


def main():
    global args
    # Handle argparse arguments
    args = parser.parse_args()
    model_path = args.model
    run_tag = args.run

    model = StarDist2D(None, name='RAW', basedir=model_path)
    PATH = args.path
    print('\n\n\nCurrent task: Cell phenotype extraction\n\n\n')
    slide_name = osp.split(PATH[:-1])[1]
    resume_path = args.resume
    save_path = args.save
    file_handler(PATH, 'PhenoExtracter')
    if run_tag != '':
        run_tag = '_' + run_tag
    tiles_path = glob(PATH + 'tiles' + run_tag + '/*.tiff')
    channel_path =  glob(PATH + '*_channels.pkl')[0]

    if save_path == '':
        save_path = PATH + 'expressions/'
        nucmask_path = PATH + 'tiles' + run_tag + '/nuc_masks/'
        for p in [save_path, nucmask_path]: os.mkdir(p) if not osp.exists(p) else None
    dapi_ch = args.dapi

    # Load channel names; for now no way to extract channel names from ome.tiff directly
    channel_dict = pickle.load(open(channel_path, 'rb'))
    dict_keys = list(channel_dict.keys())
    ch = channel_dict[dict_keys[0]][channel_dict[dict_keys[1]] == 1]
    count = 1

    # Handling a partial cell extraction
    if resume_path == '':
        gdf_global = None
    else:
        gdf_global = pd.read_csv(resume_path)
        gdf_global = gdf_global.reset_index()
    # Iterate through image tiles
    for t in tiles_path:
        print('Current image: ' + t)

        _, tile_name = osp.split(osp.splitext(osp.splitext(t)[0])[0])
        _, tile_id, x_coord, y_coord = tile_name.split('_')
        if gdf_global is not None:

            if tile_id in gdf_global['tile_id'].astype(str).values:
                count += 1
                print(f'tile {tile_id} skipped, as it was already phenotyped.')
                continue
        img = np.asarray(imread(t))
        # TMA image structure is different due to xtiff change.
        DAPI = img[dapi_ch, :, :]  # originally: DAPI = img[:, :, 0]

        gdf, labels = nuc_segmentator(DAPI, model=model, tile_id=tile_id, x_coord=x_coord, y_coord=y_coord)

        # save nuclear mask as image
        xtiff.to_tiff(labels.astype(np.uint16),  # np.transpose(img, (2, 1, 0)),
                      file=f"{nucmask_path}{tile_name}_nucmask.ome.tiff",
                      )
        if gdf.shape[0] == 0:
            continue

        # handle polygons and convert to np.arrays
        dict_raster = rasterizer(gdf, img_shape=DAPI.shape)
        kernel = np.ones((3, 3), np.uint8)

        # produce cell- and cytoplasm masks from nucleus mask
        dict_raster = raster_operations(dict_raster, kernel=kernel)
        # Handle cytoplasm overlap conflicts
        dict_raster = watershed_dilation(dict_raster)
        result = phenotyper(img, ch, gdf, dict_raster)

        geom_cols = result.columns[result.columns.str.contains('_geom')]
        result = result.drop(geom_cols, axis=1)

        # Write result
        if gdf_global is None:
            gdf_global = result
            headr = True
            result = result.reset_index(drop=True)
            result = result.loc[:, ~result.columns.duplicated()].copy()
        else:
            result = result.reset_index(drop=True)
            gdf_global = gdf_global.reset_index(drop=True)
            result = result.loc[:, ~result.columns.duplicated()].copy()
            gdf_global = gdf_global.loc[:, ~gdf_global.columns.duplicated()].copy()
            gdf_global = pd.concat([gdf_global, result], ignore_index=True)
            headr = False

        if 'Unnamed: 0' in gdf_global.columns:
            del gdf_global['Unnamed: 0']
        basename, _ = osp.splitext(t)

        result.to_csv(f'{save_path}{slide_name}{run_tag}_raw.csv',
                      mode='a', index=False, header=headr)
        print(f'{count} out of {len(tiles_path)} tiles processed.')
        count += 1


def nuc_segmentator(channel, model, tile_id, x_coord, y_coord):
    """
    Runs the StarDist segmentation algorithm.
    :param channel: Grayscale image of the target channel for nucleus segmentation. Normally, this is done on DAPI
    :param model: StarDist model
    :param tile_id: tile index, extracted from tile name
    :param x_coord: x coordinate of the tile's top left corner, extracted from tile name
    :param y_coord: y coordinate of the tile's top left corner, extracted from tile name
    :return: GeoDataFrame with shapely shapes and other morphological features.
    """
    labels, details = model.predict_instances(normalize(channel))

    # for visualization to check results.
    # plt.subplot(1, 2, 1)
    # plt.imshow(channel, cmap="gray")
    # plt.axis("off")
    # plt.title("input image")
    # plt.subplot(1, 2, 2)
    # mask = render_label(labels, img=channel)
    # plt.imshow(mask)
    # plt.axis("off")
    # plt.title("prediction + input overlay")
    # plt.show()
    polygons = np.transpose(details['coord'], axes=(0, 2, 1))
    centroids = details['points']
    num_cells = polygons.shape[0]
    print('# of Cells: ', num_cells)
    nuc_list = GeoSeries([Polygon(polygons[x, :, :]) for x in range(num_cells)])

    # Dilation of nuclear mask
    cell_list = GeoSeries([scale(Polygon(polygons[x, :, :]),
                                 xfact=np.sqrt(2), yfact=np.sqrt(2)) for x in range(num_cells)])
    cyto_list = cell_list.difference(nuc_list)
    # visu_list = cyto_list.boundary.union(nuc_list)
    centroid_list = GeoSeries([Point(centroids[x, :]) for x in range(num_cells)])
    x = np.asarray([centroids[x, 0] for x in range(num_cells)])


    x_global = x + int(y_coord)
    y = np.asarray([centroids[x, 1] for x in range(num_cells)])
    y_global = y + int(x_coord)
    geodict = {'tile_id': tile_id,
               'id': [i for i in np.arange(0, num_cells)],
               'cell_geom': cell_list,
               'nuc_geom': nuc_list,
               'cyto_geom': cyto_list,
               'centroid_geom': centroid_list,
               'centroid_x': x,
               'centroid_x_global': x_global,
               'centroid_y': y,
               'centroid_y_global': y_global,
               'nuc_area': nuc_list.area,
               'cyto_area': cyto_list.area
               }
    gdf = geopandas.GeoDataFrame(geodict)
    return gdf, labels


def rasterizer(gdf, img_shape):
    """
    Converts shapely Polygons contained within a GeoSeries to np.array rasters.
    :param gdf: GeoDataFrame with the columns 'cell' & 'nuc'
    :param img_shape: tuple of image dimensions, according to numpy_array.shape
    :return: Dictionary of arrays.
    """
    dict_raster = {}
    for name in ['cell', 'nuc']:
        if name == 'nuc':
            geom_id = ((geom, ID) for geom, ID in zip(gdf[name + '_geom'], gdf.id))
        else:
            geom_id = gdf[name + '_geom']  # object identity will not be added to raster: raster will be binary image
        raster = features.rasterize(geom_id,
                                    out_shape=(img_shape[1], img_shape[0]),
                                    all_touched=True,
                                    fill=-1,  # background value
                                    dtype=np.int16)

        # due to a coordinate mess, I need to flip and rotate the numpy array first.
        dict_raster[name] = np.rot90(np.flip(raster, axis=0), k=-1)
    return dict_raster


# This function needs some reworking: code can be cleaned up, and the epithelial implementation does not work yet.
def raster_operations(rasterdict, kernel=None):
    """
    Function to preprocess the needed images for watershedding.
    :param rasterdict: Dictionary with the marker ('nuc') and the target ('cell')
    :param kernel: Kernel used for the dilation of the target ('cell') to create the sure background ('sure_bg')
    :return: Dictionary with marker, target, and sure background.
    """
    # operations on the target image:
    rasterdict['cell'] = (rasterdict['cell'] == 1).astype(np.uint8)
    # cv2.watershed only works on rgb-targets. Therefore I will stack the image
    rasterdict['cell'] = np.stack((rasterdict['cell'], rasterdict['cell'], rasterdict['cell']))
    rasterdict['cell'] = np.transpose(rasterdict['cell'], (1, 2, 0)) * 255
    # Kernel is None when I'm doing watershed on epithelial cells. This operation does not work
    if kernel is not None:
        rasterdict['sure_bg'] = cv2.dilate(rasterdict['cell'], kernel, iterations=3)

        marker = rasterdict['nuc']
        marker = marker + 2
        marker_mask = rasterdict['sure_bg'][:, :, 0]
        marker_mask[marker > 1] = 0
        marker[marker_mask == 255] = 0
        rasterdict['nuc'] = marker
    else:
        rasterdict['nuc'] = rasterdict['nuc']
    return rasterdict


def watershed_dilation(rasterdict):
    """
    Function to apply the watershed algorithm on the target input image
    :param rasterdict: Dictionary with the marker ('nuc') and the target ('cell')
    :return: Dictionary with the marker ('nuc') and the segmentation output ('cell' & 'cyto')
    """
    segm = cv2.watershed(rasterdict['cell'].astype(np.uint8), cv2.UMat(rasterdict['nuc'].astype(np.int32)))
    segm = cv2.UMat.get(segm) - 2
    marker = rasterdict['nuc']
    rasterdict['cell'] = segm.copy()  # save the updated cell segmentation
    segm[marker > 1] = -2  # mask the nucleus region within the cell segmentation, leaving only the cytoplasm

    rasterdict['cyto'] = segm
    rasterdict['nuc'] = rasterdict['nuc'] - 2
    return rasterdict


def phenotyper(image, channels, gdf_coord, dict_raster):
    num_cell = np.max(dict_raster['nuc'] + 1)
    a = pd.DataFrame({'placeholder': np.arange(0, num_cell), 'id': np.arange(0, num_cell)})
    a = a.set_index('placeholder')
    for name in ['nuc', 'cyto']:  # removed 'cell' for file size reasons. also not really a big gain from that

        dict_pheno = {}
        for i in range(image.shape[0]):
            crop = image[i, :, :]

            dict_pheno['id'] = dict_raster[name].flatten()
            dict_pheno[channels[i] + '_' + name + '_expr'] = crop.flatten()

        df_img = pd.DataFrame(dict_pheno)
        expr_df = df_img.groupby('id').mean()
        # Check for indices that are not referring to a single cell.
        # I do not yet completely remove these indices: If I want to measure shared borders or similar in the future,
        # I will still have this option
        for i in [-3, -2, -1]:
            if i in expr_df.index:
                expr_df = expr_df.drop([i])
        a = pd.concat([a, expr_df], axis=1)
    # Result does not have the morphological columns, need to join
    # Passing 'suffixes' which cause duplicate columns {'id_total'} in the result
    # is deprecated and will raise a MergeError in a future version.
    gdf_coord = gdf_coord.set_index('id')
    result = pd.concat([gdf_coord, a], axis=1)
    return result


if __name__ == '__main__':
    main()
