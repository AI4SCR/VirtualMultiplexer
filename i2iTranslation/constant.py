import os

# Patch normalization mean and std
PATCH_NORMALIZATION_MEAN = 0.5
PATCH_NORMALIZATION_STD = 0.5

# Normalizer configuration
NORM_CFG = {
    'instance': {
        'type': 'in'
    }
}

# Number of roi classes
N_ROI_CLASSES = 2   # POS-RoI, NEG-RoI

def define_constants(
        base_path: str,
        src_marker: str,
        dst_marker: str,
        i2i_config_path: str,
        i2i_patch_size: int = 512,
        **kwargs
):
    # ********* Model inputs

    # Data split path
    DATA_SPLIT_PATH = os.path.join(base_path, 'data_splits')
    SRC_DATA_SPLIT_PATH = os.path.join(DATA_SPLIT_PATH, f'{src_marker}_splits.csv')
    DST_DATA_SPLIT_PATH = os.path.join(DATA_SPLIT_PATH, f'{dst_marker}_splits.csv')

    # Image paths
    CORE_IMAGES_PATH = os.path.join(base_path, 'cores', 'core_images')
    SRC_CORE_IMAGES_PATH = os.path.join(CORE_IMAGES_PATH, src_marker)
    DST_CORE_IMAGES_PATH = os.path.join(CORE_IMAGES_PATH, dst_marker)

    # Tissue mask paths
    CORE_TISSUE_MASKS_PATH = os.path.join(base_path, 'cores', 'core_tissue_masks')
    SRC_CORE_TISSUE_MASKS_PATH = os.path.join(CORE_TISSUE_MASKS_PATH, src_marker)
    DST_CORE_TISSUE_MASKS_PATH = os.path.join(CORE_TISSUE_MASKS_PATH, dst_marker)

    # I2I patch paths
    I2I_PATCH_SIZE = i2i_patch_size
    I2I_PATCHES_PATH = os.path.join(base_path, 'cores', 'core_i2i_patches', f'core_patches_{I2I_PATCH_SIZE}')
    SRC_I2I_PATCHES_PATH = os.path.join(I2I_PATCHES_PATH, src_marker)
    DST_I2I_PATCHES_PATH = os.path.join(I2I_PATCHES_PATH, dst_marker)

    # Path to bbox information
    SRC_BBOX_INFO_PATH = os.path.join(base_path, 'cores', 'bbox_info', src_marker)
    DST_BBOX_INFO_PATH = os.path.join(base_path, 'cores', 'bbox_info', dst_marker)

    # ********* Model outputs

    # I2I checkpoints path
    I2I_CHECKPOINTS_PATH = os.path.join(base_path, 'checkpoints', f'{src_marker}_{dst_marker}')

    # Paths for I2I prediction (AB: forward prediction, BA: reverse prediction)
    I2I_PREDICTION_PATH = os.path.join(base_path, 'predictions', f'{src_marker}_{dst_marker}', 'source_to_target')

    return {
        'base_path': base_path,
        'config_path': i2i_config_path,

        'data_split_path': DATA_SPLIT_PATH,
        'src_data_split_path': SRC_DATA_SPLIT_PATH,
        'dst_data_split_path': DST_DATA_SPLIT_PATH,

        'core_images_path': CORE_IMAGES_PATH,
        'src_core_images_path': SRC_CORE_IMAGES_PATH,
        'dst_core_images_path': DST_CORE_IMAGES_PATH,

        'core_tissue_masks_path': CORE_TISSUE_MASKS_PATH,
        'src_core_tissue_masks_path': SRC_CORE_TISSUE_MASKS_PATH,
        'dst_core_tissue_masks_path': DST_CORE_TISSUE_MASKS_PATH,

        'i2i_patches_path': I2I_PATCHES_PATH,
        'src_i2i_patches_path': SRC_I2I_PATCHES_PATH,
        'dst_i2i_patches_path': DST_I2I_PATCHES_PATH,

        'i2i_checkpoints_path': I2I_CHECKPOINTS_PATH,

        'i2i_prediction_path': I2I_PREDICTION_PATH,

        'src_bbox_info_path': SRC_BBOX_INFO_PATH,
        'dst_bbox_info_path': DST_BBOX_INFO_PATH,

        'i2i_patch_size': I2I_PATCH_SIZE,
    }