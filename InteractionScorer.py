import pandas as pd
import argparse
from glob import glob
import anndata as ad
import os.path as osp
import seaborn as sns
import numpy as np
from scipy.spatial import distance_matrix

parser = argparse.ArgumentParser(description='Calculate interaction score between epithelial and stromal cells')
parser.add_argument('path', type=str,
                    help='Path to image folder')
parser.add_argument('--tma', action='store_true', help='Flag to mark that the data is a TMA.')
parser.add_argument('-c', '--cell_type', type=str, default='Epithelium',
                    help='For which cell type will an interaction score be calculated')
parser.add_argument('-t', '--threshold', type=int, default=30,
                    help='Distance threshold for cell interaction in μm.')
parser.add_argument('-m', '--resolution', type=float, default=0.28, help='')
parser.add_argument('-r', '--run', type=str, default='', help='Tag for your current run')


def main():
    global args
    args = parser.parse_args()

    # Extract values from argparse
    PATH = args.path
    run_tag = args.run
    if run_tag != '':
        run_tag = '_' + run_tag
    slide_name = osp.split(PATH[:-1])[1]

    save_path = f'{PATH}expressions/{slide_name}{run_tag}_obs.csv'
    TMA_flag = args.tma
    thresh = args.threshold / args.resolution  # 30um / pixel resolution
    print(f'Distance radius: {thresh}px')
    cell_type = args.cell_type

    # Following cell types are excluded for the stromal interaction
    # Here I change cells to calculate the interaction score
    exclude_stromal = [cell_type, 'Noisy', 'Unknown', 'T cell', 'Macrophage', 'B cell', 'Vessel']

    file = glob(PATH + 'expressions/*_budding.h5ad')[0]
    # cell_type = ['Unknown']
    h5 = ad.read_h5ad(file)
    df = h5.obs
    # for f in file:
    #     dataH5 = ad.read_h5ad(f)
    #     dataH5.obs['ImageID'] = dataH5.obs['ImageID'].str.replace(r'-Af', '')
    #     subsetDF = dataH5.obs.loc[~dataH5.obs.GeneralType.isin(cell_type), columns]
    #     subsetDF.reset_index(inplace=True, drop=True)
    #     if df.empty:
    #         df = subsetDF
    #     else:
    #         df = pd.concat([df, subsetDF], ignore_index=True)

    # Calculate EMT score
    EMT_markers = ['Ecadherin', 'CDX2', 'bCatenin', 'Ki67']  # 'LaminB1', 'EPCAM1',
    # fibro_markers = ['FAP', 'aSMA', 'ZEB1']
    EMT_locations = ['CytoNormalized', 'NucNormalized', 'CytoNormalized', 'CytoNormalized', 'NucNormalized', 'NucNormalized']
    # fibro_locations = ['CytoNormalized', 'CytoNormalized', 'NucNormalized']
    avail_markers = [(m, l) for m, l in zip(EMT_markers, EMT_locations) if m in h5.var_names]
    #avail_markers_fibro = [(m, l) for m, l in zip(fibro_markers, fibro_locations) if m in h5.var_names]
    # for name, location in avail_markers_fibro:
    #     print(name, location)
    #     idx = np.where(h5.var.index == name)[0][0]
    #     h5.obs[name] = h5.layers[location][:, idx]
    #     h5.obs.loc[h5.obs[name] < 0, name] = 0
    #     h5.obs.loc[h5.obs[name] > 1., name] = 1.
    h5.obs['EMT_score'] = 0
    for name, location in avail_markers:
        print(name, location)
        idx = np.where(h5.var.index == name)[0][0]
        h5.obs[name] = h5.layers[location][:, idx]
        # Clip values
        h5.obs.loc[h5.obs[name] < 0, name] = 0
        h5.obs.loc[h5.obs[name] > 1., name] = 1.
        h5.obs['EMT_score'] += h5.obs[name]
    h5.obs['EMT_score'] = h5.obs['EMT_score'] / len(avail_markers)
    h5.obs.loc[h5.obs['GeneralType'] != 'Epithelium', 'EMT_score'] = 0

    if TMA_flag:
        roi_col = 'SpotID'

    else:
        roi_col = 'TileID'

    df.set_index(['ImageID', roi_col])
    image = pd.unique(df['ImageID'])[0]
    rois = pd.unique(df[roi_col])
    if not TMA_flag:
        rois = rois[rois != 0]
    interactionCols = ['stroma_interaction', 'epi_interaction', 'interaction_score']
    df[interactionCols] = 0
    # expressionCols = 'FAP_mean'
    # df[expressionCols] = 0.0

    # Calculate interaction score per roi
    for roi in rois:
        print(f'Image {image}, roi {roi}')

        # Select epithelial cells
        idx_DE = ((df[roi_col] == roi) & (df['ImageID'] == image) &
                  (df['GeneralType'] == cell_type) & (df['clusterSize'] > 0))
        # if there's no epithelial cells
        if len(df.loc[idx_DE, roi_col]) == 0:
            continue
        df_filtered = df.loc[(df[roi_col] == roi) & (df['ImageID'] == image)]

        epi_idx = df_filtered.loc[(df_filtered['GeneralType'] == cell_type) & (df['clusterSize'] > 0)].index  # .reset_index(drop=True)
        coords_epi = np.column_stack((df.loc[epi_idx, 'spatial_X'],
                                      df.loc[epi_idx, 'spatial_Y']))
        # Stromal cell selection
        if TMA_flag:
            stromal_idx = df_filtered.loc[~df_filtered['GeneralType'].isin(exclude_stromal)].index
        else:
            epi_min = (np.min(coords_epi[:, 0]), np.min(coords_epi[:, 1]))
            epi_max = (np.max(coords_epi[:, 0]), np.max(coords_epi[:, 1]))
            stromal_idx = df.loc[(~df['GeneralType'].isin(exclude_stromal)) &
                                 (df['spatial_X'] > (epi_min[0] - thresh)) &
                                 (df['spatial_Y'] > (epi_min[1] - thresh)) &
                                 (df['spatial_X'] < (epi_max[0] + thresh)) &
                                 (df['spatial_Y'] < (epi_max[1] + thresh))].index  # .reset_index(drop=True)

            intra_epi_idx = df.loc[(df['GeneralType'] == cell_type) &
                                   (df['spatial_X'] > (epi_min[0] - thresh)) &
                                   (df['spatial_Y'] > (epi_min[1] - thresh)) &
                                   (df['spatial_X'] < (epi_max[0] + thresh)) &
                                   (df['spatial_Y'] < (epi_max[1] + thresh))].index  # .reset_index(drop=True)

            coords_intra_epi = np.column_stack((df.loc[intra_epi_idx, 'spatial_X'],
                                                df.loc[intra_epi_idx, 'spatial_Y']))
        # restructure coordinates
        coords_stroma = np.column_stack((df.loc[stromal_idx, 'spatial_X'],
                                         df.loc[stromal_idx, 'spatial_Y']))

        # calculate and threshold distance matrix for interaction between epithelial and stromal cells
        dm_inter = distance_matrix(coords_stroma, coords_epi)
        am_inter = np.where(dm_inter > thresh, 0, 1)
        df.loc[idx_DE, 'stroma_interaction'] = np.sum(am_inter, axis=0)
        # dm_FAP = np.transpose(np.tile(df.loc[stromal_idx, 'FAP'], (dm_inter.shape[1], 1)))
        # am_FAP = np.where(dm_inter > thresh, np.nan, dm_FAP)
        # print(df_DE.loc[idx_DE].shape)
        # print(len(np.sum(am_inter, axis=0)))
        # FAP_mean = np.nanmean(am_FAP, axis=0)
        # df.loc[idx_DE, 'FAP_mean'] = FAP_mean


        # calculate and threshold distance matrix for interaction of epithelial cells with themselves
        if TMA_flag:
            dm_intra = distance_matrix(coords_epi, coords_epi)
        else:
            dm_intra = distance_matrix(coords_intra_epi, coords_epi)
        am_intra = np.where(dm_intra > thresh, 0, 1)
        df.loc[idx_DE, 'epi_interaction'] = np.sum(am_intra, axis=0)

        # general connectiveness score
        df.loc[idx_DE, 'interaction_score'] = df.loc[idx_DE, 'stroma_interaction'] - df.loc[
            idx_DE, 'epi_interaction']
        # Visualization:
        # plt.scatter(coords_epi[:,1], coords_epi[:,0],c=df_DE.loc[idx_DE, 'interaction_score'])
        # plt.show()
        # print(f'Mean connectiveness: {np.mean(df_DE.loc[idx_DE,"interaction_score"])}\n'
        #       f'Min connectiveness: {np.min(df_DE.loc[idx_DE,"interaction_score"])}\n'
        #       f'Max connectiveness: {np.max(df_DE.loc[idx_DE,"interaction_score"])}')
    df.to_csv(save_path, index=False)
    h5.obs = df
    h5.write_h5ad(f'{PATH}expressions/{slide_name}{run_tag}_interact.h5ad')


if __name__ == '__main__':
    main()
