import sys
import os
import numpy as np
import argparse
import openslide
import PIL.Image as Image
#import bioformats as bf
from matplotlib import pyplot as plt
#import javabridge as jb
import cv2
import imutils
from valis import registration
from valis import slide_io
from valis import preprocessing
import openslide


def main():
    slide_src_dir = "/media/mauro-gwerder/Elements/LabSat test/valis_reg"
    results_dst_dir = "./registered_images"
    registered_slide_dst_dir = "./registered_images/registered_slides"

    # Convert to ome.tiff
    # slide_io.convert_to_ome_tiff(slide_src_dir + '/AE1AE3_VIM.mrxs', slide_src_dir + '/AE1AE3_VIM.ome.tiff', 2)
    # Read slides
    # RefImage = slide_io.ImageReader(slide_src_dir + '/AE1AE3_VIM.ome.tiff')
    # RefImage.create_metadata()
    # RefImage.slide2image(6)
    # RefImage.slide2vips(6)

    # Open first slide with openslide
    # slide_A = openslide.open_slide(slide_src_dir + '/AE1AE3_VIM.mrxs')
    # img_A = mrxs_conversion(slide_A, 0, 1, dim=(20000, 25000), imgsize=(5000, 5000))
    # # plt.imshow(img_A[:, :, 2])
    # # plt.show()
    # # Open second slide with openslide
    # slide_B = openslide.open_slide(slide_src_dir + '/CD163_CDX2.mrxs')
    # img_B = mrxs_conversion(slide_B, 0, 1, dim=(20000, 25000), imgsize=(5000, 5000))
    # # plt.imshow(img_B[:, :, 2])
    # # plt.show()
    # img_overlay = np.empty((5000, 5000, 3), dtype=np.uint8)
    # img_overlay[:, :, 0] = img_A[:, :, 2]
    # img_overlay[:, :, 1] = img_B[:, :, 2]
    # plt.imshow(img_overlay)
    # plt.show()
    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(slide_src_dir, results_dst_dir, max_image_dim_px=2500,
                                   max_processed_image_dim_px=2500,
                                   thumbnail_size=5000
                                   # non_rigid_registrar_cls=None
                                   )
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    # Save all registered slides as ome.tiff
    registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap", compression="jpeg", level=1)
    #
    # Kill the JVM
    registration.kill_jvm()


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