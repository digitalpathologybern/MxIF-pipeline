# https://stackoverflow.com/questions/71083866/pandas-int64index-fix-for-futurewarning
from tifffile import imread
import numpy as np
import pandas as pd
from glob import glob
import anndata as ad
import os
import os.path as osp
import argparse
from FileHandler import *

np.random.seed(44)

parser = argparse.ArgumentParser(description='Extract Tumor bud labels from the segmentation and cell typing results')
parser.add_argument('--path', type=str, default='/media/mauro-gwerder/Lunaphore/slides/B22/',
                    help='path to image folder.')
parser.add_argument('-c', '--cell_type', type=str, default='Epithelium',
                    help='Which cell type is selected to count the cells within each segment')
parser.add_argument('-r', '--run', type=str, default='240510', help='Tag for your current run')

def main():
    global args
    args = parser.parse_args()

    # Extract values from argparse
    PATH = args.path
    run_tag = args.run
    if run_tag != '':
        run_tag = '_' + run_tag

    print('\n\n\nCurrent task: Tumor bud classification\n\n\n')
    file_handler(PATH, 'TumorBudifizer')


    cell_type = [args.cell_type]
    RESULT_PATH = PATH + 'expressions/'
    QC_PATH = PATH + 'QC/'
    TILE_PATH = glob(PATH + 'tiles' + '/masks_RFseg/*.tiff')  # PATH + 'tiles' + run_tag + '/masks_RFseg/*.tiff'
    TILE_PATH.sort()

    # Path and file handling
    ImageID = osp.split(osp.split(PATH)[0])[1]
    INSTRUCTION_NAME = glob(f'{PATH}CellType_Instructions/*.csv')[0]
    InstructionID = osp.split(osp.splitext(INSTRUCTION_NAME)[0])[1]
    print(f'{RESULT_PATH}{ImageID}_{InstructionID}.h5ad')
    dataH5 = ad.read_h5ad(f'{RESULT_PATH}{ImageID}_{InstructionID}.h5ad')

    dataH5.obs['clusterID'] = 0
    dataH5.obs['clusterSize'] = 0
    dataH5.obs['tempID'] = 0

    # Loop through tiles
    cluster_count = 0
    for TileFile in TILE_PATH:

        _, TileID = os.path.split(os.path.splitext(TileFile)[0])
        print(f'Current tile being processed: {TileFile}')
        meta = os.path.split(TileID)[1].split('_')
        segm = np.transpose(imread(TileFile), (1, 0))

        # Create tile masks (needed for overlapping cells)
        mask_x = (dataH5.obs['spatial_Y'] >= int(meta[3])) & (dataH5.obs['spatial_Y'] <= int(meta[3]) + segm.shape[0])
        mask_y = (dataH5.obs['spatial_X'] >= int(meta[2])) & (dataH5.obs['spatial_X'] <= int(meta[2]) + segm.shape[1])
        # Cells that already have cluster identity
        mask_empty = dataH5.obs['clusterID'] != 0
        mask = mask_x & mask_y

        # catches cells that are at the border and therefore are occurring in two tiles
        mask_border = mask & mask_empty
        # catches new cells
        mask_new = mask & (mask_empty == False)

        count = np.max(segm)
        segm[segm != 0] += cluster_count

        # Handling cluster IDs of clusters spanning across several tiles
        if sum(mask_border) != 0:

            new_clusters = [segm[int(Row.spatial_Y)-int(meta[3])-1, int(Row.spatial_X)-int(meta[2])-1]
                            for _, Row in dataH5.obs.loc[mask_border].iterrows()]
            dataH5.obs.loc[mask_border, 'tempID'] = new_clusters

            for new in np.unique(new_clusters):
                if new != 0:
                    new_mask = dataH5.obs['tempID'] == new
                    old_cl = np.sort(np.unique(dataH5.obs.loc[new_mask, 'clusterID']))
                    # print(old_cl)
                    if len(old_cl) > 1:
                        old_min = int(old_cl[0])
                        remaining_old = old_cl[1:]
                        mask_old = dataH5.obs['clusterID'].isin(remaining_old)
                        dataH5.obs.loc[mask_old, 'tempID'] = old_min
                    elif old_cl.size == 0:
                        #print('old_cl is empty')
                        continue
                    else:

                        old_min = int(old_cl[0])

                    segm[segm == new] = old_min
                    dataH5.obs.loc[new_mask, 'tempID'] = old_min

        dataH5.obs.loc[mask_new, 'clusterID'] = np.array([segm[int(Row.spatial_Y)-int(meta[3])-1,
                                                          int(Row.spatial_X)-int(meta[2])-1] for _, Row
                                                          in dataH5.obs.loc[mask_new].iterrows()])
        cluster_count += count

    mask = dataH5.obs['tempID'] != 0
    dataH5.obs.loc[mask, 'clusterID'] = dataH5.obs.loc[mask, 'tempID']

    genTypeLoc = (dataH5.obs['GeneralType'].isin(cell_type))
    # Counting cells per cluster
    cluster_bins = np.bincount(dataH5.obs.loc[genTypeLoc, 'clusterID'])
    dataH5.obs.loc[genTypeLoc, 'clusterSize'] = [cluster_bins[ID] for ID in dataH5.obs.loc[genTypeLoc, 'clusterID']]
    dataH5.obs.loc[dataH5.obs['clusterID'] == 0, 'clusterSize'] = 0
    dataH5.obs['CellType'] = dataH5.obs['CellType'].astype('str')
    dataH5.obs.loc[(dataH5.obs['clusterSize'].isin(np.arange(1, 5))) &
                   (dataH5.obs['CellType'].isin(cell_type)), 'CellType'] = 'Tumor bud'
    dataH5.obs.loc[(dataH5.obs['clusterSize'].isin(np.arange(5, 33))) &
                   (dataH5.obs['CellType'].isin(cell_type)), 'CellType'] = 'Poorly differentiated cluster'

    del dataH5.obs['tempID']

    dataH5.write_h5ad(f'{RESULT_PATH}{ImageID}{run_tag}_budding.h5ad')

    # Creating file for QuPath visualization
    QPdict = {'x': dataH5.obs.spatial_X,
              'y': dataH5.obs.spatial_Y,
              'class': dataH5.obs.CellType}

    QPdf = pd.DataFrame(QPdict)
    QPdf.to_csv(f'{QC_PATH}{ImageID}{run_tag}_QuPath.tsv', sep='\t', index=False)

    dataH5.obs.to_csv(f'{RESULT_PATH}{ImageID}{run_tag}_obs.csv', index=False)

    return None


if __name__ == '__main__':
    main()
