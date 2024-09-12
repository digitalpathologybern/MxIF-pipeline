import xtiff
from tifffile import imread, imwrite, TiffFile
import matplotlib.pyplot as plt
import matplotlib
import os
import javabridge as jb
import bioformats as bf
import bioformats.omexml as ome
import numpy as np
import re
from glob import glob

def main():
    jb.start_vm(class_path=bf.JARS, run_headless=True)
    bf.clear_image_reader_cache()

    PATH = '/media/mauro-gwerder/Miltenyi/Bern_hFFPE_TMA_run2/Tonsil epith/'
    imgPATHs = glob(PATH + '*.tif')
    print(f'Number of channels: {len(imgPATHs)}')
    regex_pattern = 'A-(\w+)[_.]'
    batch1 = ['DAPI', 'CD163', 'Cytokeratin', 'CD68','CD3', 'CD8a', 'CD31', 'CD20Cytoplasmic',
                 'Vimentin', 'Ki67', 'betaCatenin']
    batch2 = ['DAPI', 'CD11b', 'CD57', 'CD14', 'HLAABC', 'CD45RB', 'CD2AP', 'CD66e', 'CD45RA', 'Bcl2',
              'CD324', 'CD44', 'PlasmaCell', 'CD1c', 'CD326', 'CD147']
    batch3 = ['DAPI', 'CD4', 'Caveolin1', 'Cytokeratin14', 'Podoplanin', 'AnnexinI', 'Caldesmon', 'Cytokeratin7', 'Desmin', 'CollagenI',
              'HLADR', 'Cytokeratin5681719', 'MyosinSmoothMuscle', 'WGA', 'Actin', 'ErbB2']
    img = np.array(imread(imgPATHs[0]))
    print(np.shape(img))
    finaltiff = np.zeros((np.shape(img)[0], np.shape(img)[1], len(batch3)))
    channels = []
    idx = 0
    for i in imgPATHs:
        # use the search() method to find the first match in the string
        match = re.search(regex_pattern, i)
        captured_group = ''
        # check if a match was found
        if match:
            # extract the captured group
            captured_group = match.group(1)

            if captured_group in batch3:
                print(captured_group)
                print(f'Current channel: {captured_group}')
                channels.append(captured_group)
                img = imread(i)
                finaltiff[:, :, idx] = img
                idx += 1
        else:
            print("No match found.")

    finaltiff = finaltiff
    finaltiff = finaltiff.astype(np.uint16)
    xtiff.to_tiff(np.transpose(finaltiff, (2, 1, 0)),
                  '/media/mauro-gwerder/Miltenyi/StackedImages/TonsilEpith2.ome.tiff',
                  channel_names=channels, big_tiff=True)
    # bf.write_image('/media/mauro-gwerder/Miltenyi/CompleteImages/test.ome.tiff', finaltiff.astype(ome.PT_UINT16),
    #                pixel_type=ome.PT_UINT16, channel_names=channels
    #               )
    jb.kill_vm()



if __name__ == '__main__':
    main()