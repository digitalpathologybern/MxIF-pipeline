import pandas as pd

import numpy as np
import anndata as ad
import scanpy as sc
import scimap as sm
import os
import os.path as osp
import argparse
from glob import glob
import seaborn as sns
from FileHandler import *

sns.set(color_codes=True)

np.random.seed(44)

parser = argparse.ArgumentParser(description='Celltyping according to automated gating strategy')
parser.add_argument('path', type=str,  # default='/media/mauro-gwerder/Lunaphore/43a/',
                    help='path to image folder. Needs to contain a channel list and coordinate file in folder as well')
parser.add_argument('-s', '--savesteps', action='store_false', help='Should intermediate results be saved')
parser.add_argument('-p', '--plotting', action='store_false', help='Should intermediate results be plotted')
parser.add_argument('-t', '--threshold', type=int, default=50, help='Minimal avg DAPI expression of a cell. Needed' +
                                                                    ' for filtering false-positive cell detections')
parser.add_argument('-o', '--overlap', type=int, default=50, help='Overlap between the tiles. Needed to filter double cell predictions')
parser.add_argument('--tma', action='store_true', help='Flag to mark that the data is a TMA.')
parser.add_argument('-r', '--run', type=str, default='', help='Tag for your current run')


def main():
    global args

    # handle argparse arguments
    args = parser.parse_args()
    print('\n\n\nCurrent task: Cell-typing\n\n\n')
    PATH = args.path  # '/media/mauro-gwerder/Elements_2TB/PHD/BxTMA/expressions/'
    run_tag = args.run
    if run_tag != '':
        run_tag = '_' + run_tag
    # need file assertion statements
    expr_path = glob(f'{PATH}expressions/*{run_tag}_raw.csv')[0]
    result_path = PATH + 'expressions/'
    QC_PATH = PATH + 'QC/'
    os.mkdir(QC_PATH) if not os.path.exists(QC_PATH) else None

    file_handler(PATH, 'CellTyper')

    INSTRUCTION_NAME = glob(f'{PATH}CellType_Instructions/*.csv')[0]

    save_steps = args.savesteps  # should inbetween files be saved as well?
    plotting = args.plotting
    ImageID = osp.split(osp.split(PATH)[0])[1]
    InstructionID = osp.split(osp.splitext(INSTRUCTION_NAME)[0])[1]
    # wrong stardist predictions can arise for empty tiles.
    # This threshold removes false positives
    DAPI_THRESH = args.threshold
    TMA_FLAG = args.tma
    OVERLAP = args.overlap

    # Step 1: Data formatting
    dataDF = pd.read_csv(expr_path).dropna().drop_duplicates(['centroid_x_global', 'centroid_y_global'])
    # remove false predictions
    dataDF = dataDF.loc[dataDF['DAPI_nuc_expr'] > DAPI_THRESH].reset_index(drop=True).copy()
    dataDF = dataDF.loc[dataDF['centroid_x'] > OVERLAP].reset_index(drop=True).copy()
    dataDF = dataDF.loc[dataDF['centroid_y'] > OVERLAP].reset_index(drop=True).copy()
    cytoDF = dataDF[[col for col in dataDF.columns if 'cyto_expr' in col]]
    cytoNames = {col: col.split('_cyto')[0] for col in cytoDF.columns}
    cytoDF.rename(cytoNames, inplace=True, axis=1)
    cytoDF = cytoDF.drop(['DAPI'], axis=1)

    # Changing colnames. to do: change names prior in pipeline
    if TMA_FLAG:
        obsCols = ['centroid_x_global', 'centroid_y_global', 'nuc_area', 'spot_id', 'id']
        obsNames = {'centroid_x_global': 'spatial_Y', 'centroid_y_global': 'spatial_X', 'nuc_area': 'Area',
                    'spot_id': 'SpotID', 'id': 'CellID'}
    else:
        obsCols = ['centroid_x_global', 'centroid_y_global', 'nuc_area', 'id', 'tile_id']
        obsNames = {'centroid_x_global': 'spatial_Y', 'centroid_y_global': 'spatial_X', 'nuc_area': 'Area',
                    'id': 'CellID', 'tile_id': 'TileID'}
    obsDF = dataDF[obsCols]
    obsDF['ImageID'] = ImageID
    obsDF.rename(columns=obsNames, inplace=True)

    dataH5 = ad.AnnData(cytoDF)
    dataH5.obs = obsDF
    if save_steps:
        dataH5.write_h5ad(f'{result_path}{ImageID}{run_tag}.h5ad')
    # Workaround: Indices of the df need to be categorical 'object' for downstream function
    dataIdx = pd.Index(list(dataH5.obs.index))
    dataH5.obs.index = dataIdx.astype('object')

    # Step 2: quality control. Taken from Akoya webinar tutorial
    SNRvalues = compute_top20_btm10(dataH5)
    SNRvalues['StainQuality'] = 'sufficient'
    SNRvalues.loc[SNRvalues['top20btm10'] < 10.0, 'StainQuality'] = 'insufficient'
    SNRvalues.to_csv(f'{QC_PATH}{ImageID}{run_tag}_SNR.csv')

    # Step 3: Data rescaling per channel. Taken from Akoya webinar tutorial
    dataH5 = removeoutliers(dataH5)
    # Also create a layer where the data is quantile normalized
    dataH5.layers['CytoNormalized'] = quantnorm(dataH5.X)
    # nuclear channels inclusion as another layer
    nucDF = dataDF[[col for col in dataDF.columns if 'nuc_expr' in col]]
    nucNames = {col: col.split('_nuc')[0] for col in nucDF.columns}
    nucDF.rename(nucNames, inplace=True, axis=1)
    nucDF = nucDF.drop(['DAPI'], axis=1)
    nucH5 = ad.AnnData(nucDF)
    dataH5.layers['NucNormalized'] = quantnorm(nucH5.X)

    dataH5 = sm.pp.rescale(dataH5, imageid='ImageID', method='by_image')
    if save_steps:
        dataH5.write_h5ad(f'{result_path}{ImageID}{run_tag}_rescaled.h5ad')

    # dataH5 = sc.read_h5ad(f'{RESULT_PATH}{ImageID}_rescaled.h5ad')
    instructDF = pd.read_csv(INSTRUCTION_NAME)

    # Step 4: Gating according to instruction file.
    sm.tl.phenotype_cells(dataH5, instructDF.drop('GeneralType', axis=1), label='CellType')
    # Summary of cell types. Part of the quality control
    CT_summary = dataH5.obs.CellType.value_counts()
    pd.DataFrame(CT_summary).to_csv(f'{QC_PATH}{ImageID}{run_tag}_CTsummary.csv')
    # Save x and y coordinates in scanpy's obsm for easy plotting
    dataH5.obsm['X_spatial'] = dataH5.obs[['spatial_X', 'spatial_Y']].values
    dataH5.obs = dataH5.obs.merge(instructDF[['CellType', 'GeneralType']], on='CellType', how='left')
    dataH5.obs.loc[dataH5.obs.CellType == 'Unknown', 'GeneralType'] = 'Unknown'
    dataH5.write_h5ad(f'{result_path}{ImageID}{run_tag}_{InstructionID}.h5ad')
    dataH5.obs.to_csv(f'{PATH}expressions/{ImageID}{run_tag}_obs_CT.csv')
    if plotting:
        markers = instructDF.columns[3:len(instructDF.columns)]
        sc.pl.heatmap(dataH5, markers, groupby='CellType', swap_axes=True, standard_scale='var',
                      save=f'_{ImageID}{run_tag}_{InstructionID}.png', show=False)
        sc.pl.matrixplot(dataH5, markers, groupby='CellType', swap_axes=True, standard_scale='var',
                         save=f'{ImageID}{run_tag}_{InstructionID}.png', show=False)

    return None


# Function to compute SNR for each protein in each sample
def compute_top20_btm10(andat):
    """
    Compute the ratio of top 20th percentile to bottom 10th percentile for each protein in each sample
    Input: anndata object
    Output: a dataframe with sampleID, Protein, ratio of top20/btm10
    """
    top20btm10DF = pd.DataFrame(columns=['ImageID', 'Protein', 'top20btm10'])
    # for each sample
    for sID in andat.obs.ImageID.sort_values().unique():
        subAD = andat[andat.obs.ImageID == sID]
        for x in subAD.var_names:
            aX = subAD[:, x].X.flatten()  # all expression values of the current marker
            # compute 20 largest values in aX
            top20 = np.sort(aX)[-20:]
            # compute the mean of bottom 10th percentile of aX
            btm10 = np.sort(aX)[:int(len(aX) * 0.1)]
            top20btm10 = np.mean(top20) / np.mean(btm10)
            top20btm10DF = top20btm10DF.append({'ImageID': sID, 'Protein': x, 'top20btm10': top20btm10},
                                               ignore_index=True)
    return top20btm10DF


# for each marker clip value to mean of top 20 values
def mediantop20(subad):
    outAD = subad.copy()
    for ix, x in enumerate(outAD.var_names):
        aX = subad[:, x].X.flatten()
        top20 = np.sort(aX)[-20:]
        outAD.X[:, ix] = np.clip(subad[:, x].X.flatten(), 0, np.median(top20))
    return outAD


# remove expression outliers from the data
def removeoutliers(andat):

    # separate each sample
    s = {}
    for sID in andat.obs.ImageID.sort_values().unique():
        s[sID] = andat[andat.obs['ImageID'] == sID]
        s[sID] = mediantop20(s[sID])
    outAD = sc.concat(s.values())
    return outAD


def quantnorm(df, minval=0.001, q_low=0.01, q_high=0.996):
    # Quantized normalization
    res = minval + df
    res = np.log10(res)
    quants = np.nanquantile(res, q=[q_low, q_high], axis=0)
    quants[quants == 0] = 0.0001
    res = (res - quants[0, :]) / (quants[1, :] - quants[0, :])
    return res


if __name__ == '__main__':
    main()
