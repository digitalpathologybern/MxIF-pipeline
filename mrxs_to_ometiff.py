#from valis import slide_io
import openslide
import bioformats as bf
import javabridge as jb
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    jb.start_vm(class_path=bf.JARS, )
    PATH = '/media/mauro-gwerder/Elements/LabSat test/valis_reg/'
    SAVE_PATH = './ome_tiffs_save/'
    slide = openslide.OpenSlide(PATH + 'AE1AE3_VIM.mrxs')
    img = mrxs_conversion(slide, 0, 1, dim=(20000, 25000), imgsize=(5000, 5000))
    bf.write_image(SAVE_PATH + 'AE1AE3_VIM_Crop.ome.tiff', img, bf.PT_UINT8)
    # plt.imshow(img)
    # plt.show()
    jb.kill_vm()


def mrxs_conversion(slide, reading_level=1, extraction_level=6, dim=None, imgsize=None):
    x, y = slide.properties[openslide.PROPERTY_NAME_BOUNDS_X], slide.properties[
        openslide.PROPERTY_NAME_BOUNDS_Y]
    if dim is None:
        dim = (int(int(x)), int(int(y)))
    coord = (dim[0] + int(int(x)),
             dim[1] + int(int(y))
             )
    w, h = slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH], slide.properties[
        openslide.PROPERTY_NAME_BOUNDS_HEIGHT]
    if imgsize is None:
        imgsize = (int(int(w) / 2 ** reading_level), int(int(h) / 2 ** reading_level))
        #wh_orig = (int(int(w) / 2 ** extraction_level), int(int(h) / 2 ** extraction_level))
    rgb_im = slide.read_region(coord, extraction_level, imgsize)
    rgb_im = np.array(rgb_im)
    rgb_im[rgb_im[:, :, 3] != 255] = 255
    rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGBA2RGB)
    return rgb_im


if __name__ == '__main__':
    main()


