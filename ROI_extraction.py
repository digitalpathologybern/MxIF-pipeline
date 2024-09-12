import bioformats as bf
# Javabridge needs newer numpy version when this error pops up.
# numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
# Solution: pip install --upgrade numpy
import numpy as np
import xtiff
from glob import glob
import pickle
import javabridge as jb
import argparse
import os
import os.path as osp
from FileHandler import *


# Defines new argparse type, needed for roi extraction
def xywh(s):
    try:
        x, y, w, h = map(int, s.split(','))
        return x, y, w, h
    except Exception:
        raise argparse.ArgumentTypeError("Coordinates must be in x,y,w,h format")


np.random.seed(44)

parser = argparse.ArgumentParser(description='load data')
parser.add_argument('path', type=str,
                    help='path to image folder. Needs to contain a channel list and coordinate file in folder as well')
parser.add_argument('-x', '--XYWH', type=xywh, default='0,0,0,0')
# only works for OME tiffs for now.
# parser.add_argument('-f', '--format', default='OME', choices=['OME', 'mrxs'])
parser.add_argument('-s', '--save', type=str, default='')
parser.add_argument('-o', '--overlap', type=int, default=40,
                    help='Adds an overlap to the tiles for x pixels in each direction')
parser.add_argument('-t', '--num_tiles', type=int, default=21, help='number of tiles per dimension.')
parser.add_argument('-r', '--run', type=str, default='', help='Tag for your current run')
# parser.add_argument('--fraction', type=float, default=0.0,
#                     help='Specifies the fraction of panCK area on a tile to be used for analysis')


def main():
    global args
    args = parser.parse_args()
    # Start image extraction via javabridge
    jb.start_vm(class_path=bf.JARS, run_headless=True)
    bf.clear_image_reader_cache()
    roi_coords = args.XYWH
    overlap = args.overlap
    PATH = args.path
    run_tag = args.run

    slide_name = osp.split(PATH[:-1])[1]
    print('\n\n\nCurrent task: WSI tile extraction\n\n\n')
    file_handler(PATH, 'ROI_extraction')

    img_path = glob(PATH + '*.ome.tiff')[0]
    channel_path = glob(PATH + '*_channels.pkl')[0]
    save_path = args.save
    if save_path == '':
        save_path = PATH

    if run_tag != '':
        run_tag = '_' + run_tag

    tilePATH = save_path + 'tiles' + run_tag + '/'
    os.mkdir(tilePATH) if not osp.exists(tilePATH) else None

    # Load channel names; for now no way to extract channel names from ome.tiff directly
    channel_dict = pickle.load(open(channel_path, 'rb'))
    dict_keys = list(channel_dict.keys())
    skip_channels = np.where(channel_dict[dict_keys[1]] == 0)
    new_channelnames = channel_dict[dict_keys[0]][channel_dict[dict_keys[1]] == 1]

    # Extract metadata
    metadata = bf.OMEXML(bf.get_omexml_metadata(img_path))
    nXpixels = metadata.image().Pixels.get_SizeX()
    nYpixels = metadata.image().Pixels.get_SizeY()
    nChannels = metadata.image().Pixels.get_SizeT()

    num_tiles = args.num_tiles
    # Calculate tile coordinates
    if sum(roi_coords) == 0:
        x_boxsize = int(np.floor((nXpixels-overlap) / num_tiles))
        y_boxsize = int(np.round((nYpixels-overlap) / num_tiles))

        print('Tile-size: ', x_boxsize, 'x', y_boxsize)

        w = x_boxsize + overlap  # increases width by the dilation in both directions.
        h = y_boxsize + overlap  # increases width by the dilation in both directions.

        x_arr = np.arange(0, nXpixels-overlap, x_boxsize)[:-1]
        y_arr = np.arange(0, nYpixels-overlap, y_boxsize)[:-1]
    else:
        x, y, w, h = roi_coords
        x_arr = [x]
        y_arr = [y]

    for i, y in enumerate(y_arr):
        for j, x in enumerate(x_arr):
            index = i * num_tiles + j

            w = min(nXpixels - x, w)
            h = min(nYpixels - y, h)

            outStack = np.empty((len(new_channelnames), h, w))

            tilename = f'{tilePATH}{slide_name}_{str(index)}_{str(x)}_{str(y)}.ome.tiff'
            if osp.exists(tilename):
                continue
            for ch in range(nChannels):
                if ch in skip_channels:
                    continue
                with bf.ImageReader(img_path) as reader:
                    img = reader.read(t=ch, XYWH=(x, y, w, h), rescale=False)

                outStack[ch, :, :] = img
            finaltiff = outStack.astype(np.uint16)
            xtiff.to_tiff(finaltiff,  # np.transpose(img, (2, 1, 0)),
                          file=tilename,
                          channel_names=new_channelnames, big_tiff=True)

    jb.kill_vm()


if __name__ == '__main__':
    main()
