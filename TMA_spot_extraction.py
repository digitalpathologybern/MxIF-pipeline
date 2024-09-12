import xtiff
import os
import os.path as osp
import javabridge as jb
import bioformats as bf
import numpy as np
import pandas as pd
import argparse
import pickle
from glob import glob
from FileHandler import *

np.random.seed(44)

parser = argparse.ArgumentParser(description='load data')
parser.add_argument('path', type=str,
                    help='path to image folder. Needs to contain a channel list and coordinate file in folder as well')
parser.add_argument('-p', '--print_info', action='store_true')
parser.add_argument('-s', '--save', type=str, default='')
parser.add_argument('--diameter', type=int, default=1100, help='Diameter of one TMA core')
parser.add_argument('-o', '--overlap', type=int, default=30,
                    help='Adds an overlap to the tiles for x pixels in each direction')
parser.add_argument('-r', '--run', type=str, default='', help='Tag for your current run')


def main():
    global args
    args = parser.parse_args()

    # Start image extraction via javabridge
    jb.start_vm(class_path=bf.JARS, run_headless=True)
    bf.clear_image_reader_cache()
    PATH = args.path
    run_tag = args.run
    print('\n\n\nCurrent task: TMA spot extraction\n\n\n')
    file_handler(PATH, 'TMA_spot_extraction')

    img_path = glob(PATH + '*.ome.tiff')[0]
    csv_path = glob(PATH + '*.csv')[0]
    channel_path = glob(PATH + '*.pkl')[0]
    slidename = osp.split(PATH[:-1])[1]

    savePATH = args.save
    if savePATH == '':
        savePATH = PATH
    if run_tag != '':
        run_tag = '_' + run_tag
    tilePATH = 'savePATH' + 'tiles' + run_tag + '/'
    os.mkdir(tilePATH) if not os.path.exists(tilePATH) else None

    SPOT_DIAM = args.diameter  # spot diameter in um

    # Extract metadata
    metadata = bf.OMEXML(bf.get_omexml_metadata(img_path))
    nXpixels = metadata.image().Pixels.get_SizeX()
    nYpixels = metadata.image().Pixels.get_SizeY()
    nChannels = metadata.image().Pixels.get_SizeT()
    resolution = metadata.image().Pixels.get_PhysicalSizeX()  # pixel resolution

    # Load channel names; for now no way to extract channel names from ome.tiff directly
    channel_dict = pickle.load(open(channel_path, 'rb'))
    dict_keys = list(channel_dict.keys())
    skip_channels = np.where(channel_dict[dict_keys[1]] == 0)
    new_channelnames = channel_dict[dict_keys[0]][channel_dict[dict_keys[1]] == 1]
    # Get spot_size to a nice number
    spot_size = int(round(SPOT_DIAM / resolution)) - 3
    overlap = args.overlap  # pixels of overlap
    tile_size = int(spot_size/2) + overlap

    # need to improve file structure such that it can be compared to filename.
    arrdt = pd.read_csv(csv_path, index_col='Unique_ID')
    arrdt = arrdt.dropna()
    arrdt = arrdt.astype({'Centroid_x_px': 'int', 'Centroid_y_px': 'int'})
    slidedt = arrdt.loc[arrdt['ImageID'] == slidename]
    slidedt = slidedt.reset_index()

    tile_count = 1
    # Handle tile size for each core
    for r in range(len(slidedt['Centroid_x_px'])):
        x_0 = max(0, slidedt['Centroid_x_px'][r] - int(round(spot_size / 2)))
        x_1 = x_0 + tile_size - (overlap * 2)
        x_spot = np.array([x_0, x_1, x_0, x_1])
        y_0 = max(0, slidedt['Centroid_y_px'][r] - int(round(spot_size / 2)))
        y_1 = y_0 + tile_size - (overlap * 2)
        y_spot = np.array([y_0, y_0, y_1, y_1])
        w_spot = np.min(np.array([np.repeat(tile_size, 4), nXpixels - x_spot]), axis=0)
        h_spot = np.min(np.array([np.repeat(tile_size, 4), nYpixels - y_spot]), axis=0)
        spot_name = slidedt['SpotID'][r]

        # Extract tiles
        with bf.ImageReader(img_path) as reader:
            for i in range(4):
                finaltiff = np.empty((len(new_channelnames), h_spot[i], w_spot[i]))
                current_channel = 0
                for ch in range(nChannels):
                    if ch in skip_channels:
                        continue

                    img = reader.read(t=ch, XYWH=(x_spot[i], y_spot[i], w_spot[i], h_spot[i]), rescale=False)
                    finaltiff[current_channel, :, :] = img
                    current_channel += 1
                finaltiff = finaltiff.astype(np.uint16)
                xtiff.to_tiff(finaltiff,  # np.transpose(img, (2, 1, 0)),
                              file=f"{tilePATH}{slidename}_{spot_name}-{tile_count}_{x_spot[i]}_{y_spot[i]}.ome.tiff",
                              channel_names=new_channelnames, big_tiff=True)
                tile_count += 1
    jb.kill_vm()


if __name__ == '__main__':
    main()
