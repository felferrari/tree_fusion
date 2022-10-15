import argparse
import pathlib
from conf import paths, default, general
import numpy as np
from osgeo import gdal, gdalconst
import os
from utils.ops import load_opt_image, load_label_image, save_dict
from matplotlib import pyplot as plt
from skimage.util import view_as_windows


parser = argparse.ArgumentParser(
    description='prepare the files to be used in the training/testing steps'
)

parser.add_argument( # image
    '-i', '--image',
    type = pathlib.Path,
    default = paths.PATH_IMG,
    help = 'Path to the source image (.tif) with all bands'
)

parser.add_argument( # train data
    '--train',
    type = pathlib.Path,
    default = paths.PATH_TRAIN_LABEL,
    help = 'Path to the train data (.tif)'
)

parser.add_argument( # test data
    '--test',
    type = pathlib.Path,
    default = paths.PATH_TEST_LABEL,
    help = 'Path to the test data (.tif)'
)

parser.add_argument( # min target class pixels
    '-m', '--min-target-classes',
    type = float,
    default = default.MIN_TRAIN_CLASS,
    help = 'Minimum proportion of target pixels classes [0-1]'
)

parser.add_argument( # validation proportion split
    '-V', '--validation-split',
    type = float,
    default = default.VAL_SPLIT,
    help = 'Validation proportion to split [0-1]'
)

args = parser.parse_args()

if not os.path.exists(paths.PREPARED_PATH):
    os.mkdir(paths.PREPARED_PATH)


img = load_opt_image(str(args.image))
train_label = load_label_image(str(args.train))
test_label = load_label_image(str(args.test))
shape = train_label.shape

#remove classes
for r_class_id in general.REMOVED_CLASSES:
    train_label[train_label==r_class_id] = 99
    test_label[test_label==r_class_id] = 99

#remap labels
uni = np.unique(train_label)
new_labels = np.arange(uni.shape[0])
remap_dict = dict(zip(new_labels, uni))

train_label_remap = np.empty_like(train_label, dtype=np.uint8)
test_label_remap = np.empty_like(test_label, dtype=np.uint8)

for dest_key, source_key in remap_dict.items():
    train_idx = train_label == source_key
    train_label_remap[train_idx] = dest_key

    test_idx = test_label == source_key
    test_label_remap[test_idx] = dest_key

save_dict(remap_dict, os.path.join(paths.PREPARED_PATH, 'map.data'))

del train_label, test_label

patch_size = general.PATCH_SIZE
train_step = int((1-general.PATCH_OVERLAP)*patch_size)


idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

label_patches = view_as_windows(train_label_remap, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))

keep_patches = np.mean(np.logical_and((label_patches != 0), (label_patches != general.DISCARDED_CLASS)), axis=(1,2)) >= args.min_target_classes

idx_patches = idx_patches[keep_patches]
np.random.seed(123)
np.random.shuffle(idx_patches)
n_patches = idx_patches.shape[0]
n_val = int(args.validation_split * n_patches)
train_idx_patches = idx_patches[n_val:]
val_idx_patches = idx_patches[:n_val]

np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'), img[:,:,:general.N_OPTICAL_BANDS])
np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LIDAR}_img.npy'), img[:,:,general.N_OPTICAL_BANDS:])
np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy'), train_label_remap)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_test.npy'), test_label_remap)

np.save(os.path.join(paths.PREPARED_PATH, f'train_patches.npy'), train_idx_patches)
np.save(os.path.join(paths.PREPARED_PATH, f'val_patches.npy'), val_idx_patches)


base_data = gdal.Open(str(args.image), gdalconst.GA_ReadOnly)

geo_transform = base_data.GetGeoTransform()
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize
crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

output = os.path.join(paths.PREPARED_PATH, f'test_label.tif')

target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(geo_transform)
target_ds.SetSpatialRef(crs)
target_ds.SetProjection(proj)

target_ds.GetRasterBand(1).WriteArray(test_label_remap)
target_ds = None