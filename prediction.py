import argparse
import pathlib
import importlib
from conf import default, general, paths
import os
import time
import sys
from utils.dataloader import TreePredDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from osgeo import ogr, gdal, gdalconst
from utils.ops import load_dict

parser = argparse.ArgumentParser(
    description='Train NUMBER_MODELS models based in the same parameters'
)

parser.add_argument( # Experiment number
    '-e', '--experiment',
    type = int,
    default = 1,
    help = 'The number of the experiment'
)

parser.add_argument( # batch size
    '-b', '--batch-size',
    type = int,
    default = default.PREDICTION_BATCH_SIZE,
    help = 'The number of samples of each batch'
)

parser.add_argument( # Number of models to be trained
    '-n', '--number-models',
    type = int,
    default = 1,
    help = 'The number models to be trained from the scratch'
)

parser.add_argument( # Experiment path
    '-x', '--experiments-path',
    type = pathlib.Path,
    default = paths.PATH_EXPERIMENTS,
    help = 'The patch to data generated by all experiments'
)

parser.add_argument( # Base image to generate geotiff pred
    '-i', '--base-image',
    type = pathlib.Path,
    default = paths.PATH_IMG,
    help = 'The patch to base image to generate Geotiff prediction'
)

args = parser.parse_args()

exp_path = os.path.join(str(args.experiments_path), f'exp_{args.experiment}')
logs_path = os.path.join(exp_path, f'logs')
models_path = os.path.join(exp_path, f'models')
visual_path = os.path.join(exp_path, f'visual')
predicted_path = os.path.join(exp_path, f'predicted')
results_path = os.path.join(exp_path, f'results')


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.set_num_threads(10)

model_m =importlib.import_module(f'conf.model_{args.experiment}')
model = model_m.get_model()
model.to(device)

model_path = os.path.join(models_path, 'model.pt')
model.load_state_dict(torch.load(model_path))


dataset = TreePredDataSet(device=device)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

outfile = os.path.join(logs_path, f'pred_{args.experiment}.txt')
overlaps = general.PREDICTION_OVERLAPS
with open(outfile, 'w') as sys.stdout:
    pred_global_sum = np.zeros(dataset.original_shape+(general.N_CLASSES,))
    for overlap in overlaps:
        dataset = TreePredDataSet(device=device, overlap = overlap)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        t0 = time.perf_counter()
        pbar = tqdm(dataloader)
        preds = None
        for X in pbar:
            with torch.no_grad():
                pred = model(X).to('cpu')
            if preds is None:
                preds = pred#.to('cpu')
            else:
                preds = torch.cat((preds, pred), dim=0)
        #preds = preds.view((dataset.original_blocks_shape)).numpy().astype(np.uint8)
        preds = np.moveaxis(preds.numpy().astype(np.uint8), 1, -1)
        pred_sum = np.zeros(dataset.padded_shape+(general.N_CLASSES,)).reshape((-1, general.N_CLASSES))
        pred_count = np.zeros(dataset.padded_shape+(general.N_CLASSES,)).reshape((-1, general.N_CLASSES))
        one_window = np.ones((general.PATCH_SIZE, general.PATCH_SIZE, general.N_CLASSES))
        for idx, idx_patch in enumerate(tqdm(dataset.idx_patches)):
            pred_sum[idx_patch] += preds[idx]
            pred_count[idx_patch] += one_window
        pred_sum = pred_sum.reshape(dataset.padded_shape+(general.N_CLASSES,))
        pred_count = pred_count.reshape(dataset.padded_shape+(general.N_CLASSES,))

        pred_sum = pred_sum[general.PATCH_SIZE:-general.PATCH_SIZE,general.PATCH_SIZE:-general.PATCH_SIZE,:]
        pred_count = pred_count[general.PATCH_SIZE:-general.PATCH_SIZE,general.PATCH_SIZE:-general.PATCH_SIZE,:]

        pred_global_sum += pred_sum / pred_count

    pred_global = pred_global_sum / len(overlaps)
    pred_b = pred_global.argmax(axis=-1).astype(np.uint8)

    np.save(os.path.join(predicted_path, 'pred.npy'), pred_b)

    remap_dict = load_dict(os.path.join(paths.PREPARED_PATH, 'map.data'))

    pred_b_remaped = np.empty_like(pred_b)
    for dest, source in remap_dict.items():
        print(f'{dest}-{source}')
        pred_b_remaped[pred_b==dest] = source


    base_data = gdal.Open(str(args.base_image), gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()

    output = os.path.join(predicted_path, f'pred.tif')

    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetSpatialRef(crs)
    target_ds.SetProjection(proj)

    target_ds.GetRasterBand(1).WriteArray(pred_b_remaped)
    target_ds = None


    print('Done')

        