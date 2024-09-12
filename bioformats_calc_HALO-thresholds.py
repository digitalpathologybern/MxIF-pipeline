from PIL import Image
import bioformats as bf
import json
import numpy as np
import javabridge as jb


def main():
    jb.start_vm(class_path=bf.JARS)

    # path_compressed = "/media/mauro-gwerder/Elements/PHD/images/31b/B12.35816_subtractedAF_compressedJPEG.ome.tiff"
    path_compressed = "/media/mauro-gwerder/Elements/PHD/images/31a/compressed-JPEG90/20220307_144634_1_wTi5JQ_COM13_01_31a_20220702_B07.ome.tiff"
    #path_orig = '/media/mauro-gwerder/Elements/PHD/images/31a/export-17plex+singleDAPI_backgroundsubtraction/20220307_COM13_01_31a_B07.21461-IA.ome.tiff'
    metadata = bf.OMEXML(bf.get_omexml_metadata(path_compressed))
    print(metadata.image().ns)
    nTimepoints = metadata.image().Pixels.get_SizeT()
    nChannels = metadata.image().Pixels.get_SizeC()
    nXpixels = metadata.image().Pixels.get_SizeX()
    nYpixels = metadata.image().Pixels.get_SizeY()
    pixelType = metadata.image().Pixels.get_PixelType()
    print(f'BIOFORMATS-METADATA:\nTimepoints: {nTimepoints}\nChannels:{nChannels}\n' +
          f'Pixels in X dimension:{nXpixels}\nPixels in Y dimension:{nYpixels}\nPixel type:{pixelType}')
    # Create empty array to fill with data from file
    outStack = np.empty((1, nYpixels, nXpixels), dtype=pixelType)
    channel_names = ['DAPI', 'Bcatenin', 'CD20', 'aSMA', 'CD68', 'CD47', 'CDX2', 'CD8', 'CD3', 'ZEB1', 'CD90',
                     'CD163', 'FAP', 'TAGLN', 'Vimentin', 'CK', 'Ecadherin', 'MYL9']
    channel_index = [0, 3, 4, 7, 8, 11, 13, 15, 16, 19, 21, 23, 25, 26, 29, 30, 33, 34]
    # Read frames in to outStack
    means = []
    with open('histo_measures_channels_31a_comp.csv', 'w') as file:
        file.write('channel,mean,5th_quant,25th_quant,50th_quant,75th_quant,90th_quant,95th_quant,99th_quant\n')
        with bf.ImageReader(path_compressed) as reader:
            for ch_ind, ch_name in zip(channel_index, channel_names):
                # print(ch_name, ch_ind)
                outStack[0] = reader.read(t=ch_ind, rescale=False)
                channel_mean = np.mean(outStack[0].flatten()) * 256
                channel_quant = np.quantile(outStack[0].flatten(), [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]) * 256
                file.write(
                    f'{ch_name},{channel_mean:.3f},{channel_quant[0]},{channel_quant[1]},{channel_quant[2]},{channel_quant[3]},{channel_quant[4]},{channel_quant[5]},{channel_quant[6]}\n')
            # outStack[t] = reader.read(t=t+7, rescale=False)
            # print(np.max(outStack[t]))
            means.append(channel_mean)
            reader.close()
        print('reading is done')
        # print('all channel means: ', means)

    # plt.imshow(outStack[ch, 23000:33400, 2000:12400], cmap='viridis')
    # plt.show()

    jb.kill_vm()


if __name__ == '__main__':
    main()
