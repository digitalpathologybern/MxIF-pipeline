from glob import glob
import os.path as osp
# import argparse

# parser = argparse.ArgumentParser(description='At each stage of the MxIF pipeline, check if all files are available'
#                                              'and there is no ambiguity regarding the file-naming')
# parser.add_argument('--path', type=str,  default='/media/mauro-gwerder/Elements_2TB/PHD/COMET/B08/',
#                     help='Which path to search for files in')
# parser.add_argument('-s', '--script', type=str, default='ROI_extraction',
#                     help='For which script should files be checked.')


def file_handler(path, script):
    # global args
    # args = parser.parse_args()
    # Defines what files are needed for each script
    if script == 'ROI_extraction':
        folders = ['', '']
        extensions = ['.ome.tiff', '.pkl']
        tags = ['', '_channels']
    elif script == 'TMA_spot_extraction':
        folders = ['', '', '']
        extensions = ['.ome.tiff', '.csv', '.pkl']
        tags = ['', '', '_channels']
    elif script == 'PhenoExtracter':
        folders = ['']
        extensions = ['.pkl']
        tags = ['_channels']
        path_to_assert = path + '/tiles/'
        assert osp.exists(path_to_assert), f'There exists no target "tiles" folder. Please run the tile extraction' \
                                         f'first or properly link the "tiles" folder Path: {path_to_assert}'
    elif script == 'EpiSegmentor':
        folders = ['','']
        extensions = ['.pkl','.pkl']
        tags = ['_channels','_model']
        assert osp.exists(path + '/tiles/'), 'There exists no target "tiles" folder. Please run the tile extraction' \
                                            'first or properly link the "tiles" folder'
    elif script == 'CellTyper':
        folders = ['/expressions','/CellType_Instructions']
        extensions = ['.csv', '.csv']
        tags = ['_raw','']
    elif script == 'TumorBudifizer':
        folders = ['/CellType_Instructions']
        extensions = ['.csv']
        tags = ['']
        assert osp.exists(path + '/tiles/masks_RFseg/'), 'There exists no target "tiles" folder. Please run the tile extraction ' \
                                            'first or properly link the "tiles" folder'
    for folder, extension, tag in zip(folders, extensions, tags):
        assure_files(path + folder, extension, tag=tag)


def assure_files(path, extension, tag=''):
    """
    Assess uniqueness of file path.s
    :param path: path to file
    :param extension: file extension specification
    :param tag: Unique tag to differentiate from files with the same extension
    :return: None (at least for now)
    """
    print(f'{path}/*{tag}{extension}')
    paths = glob(f'{path}/*{tag}{extension}')

    assert len(paths) == 1, f'Ambiguous file naming: instead of one file, {len(paths)}' \
                            f' files of type {extension} and tag {tag} were found in target folder'
    return None
