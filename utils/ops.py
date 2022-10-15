import json
import numpy as np
import os
import sys
from osgeo import gdal_array
import pickle

def load_json(fp):
    with open(fp) as f:
        return json.load(f)


def save_json(dict_: dict, path_to_file: str):
    """Save a dictionary to a file path.

    Args:
        dict_ (dict): dictionary to be saved.
        path_to_file (str): Path to file.
    """
    with open(path_to_file, 'w') as f:
        json.dump(dict_, f, indent=4)

def save_dict(dict_: dict, path_to_file: str):
    with open(path_to_file, 'wb') as f:
        pickle.dump(dict_, f)

def load_dict(path_to_file: str):
    with open(path_to_file, 'rb') as f:
        return pickle.load(f)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_opt_image(patch):
    # Read tiff Image
    #print (patch)
    #img_tif = TIFF.open(patch)
    #img = img_tif.read_image()
    img = gdal_array.LoadFile(patch)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)


def load_label_image(patch):
    img = gdal_array.LoadFile(patch)
    return img


def load_SAR_image(patch):
    '''Function to read SAR images'''
    db_img = gdal_array.LoadFile(patch)
    temp_dn_img = 10**(db_img/10)
    temp_dn_img[temp_dn_img > 1] = 1
    return np.moveaxis(temp_dn_img, 0, -1)


def load_SAR_DN_image(patch):
    '''Function to read SAR images'''
    im = gdal_array.LoadFile(patch)
    return np.expand_dims(im, axis=-1)


def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)] = 0  # Filter NaN values.
    if len(mask) == 1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask != 2, band].ravel(
        ), bins=bins)  # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist < uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist < bth])])/100
        img[:, :, band][img[:, :, band] > max_value] = max_value
        img[:, :, band][img[:, :, band] < min_value] = min_value
    return img


def create_exps_paths(exp_n):
    exps_path = 'exps'

    exp_path = os.path.join(exps_path, f'exp_{exp_n}')
    models_path = os.path.join(exp_path, 'models')

    results_path = os.path.join(exp_path, 'results')
    predictions_path = os.path.join(results_path, 'predictions')
    visual_path = os.path.join(results_path, 'visual')

    logs_path = os.path.join(exp_path, 'logs')

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    return exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path


def load_exp(exp_n=None):
    if exp_n is None:
        if len(sys.argv) == 1:
            return None
        else:
            return load_json(os.path.join('conf', 'exps', f'exp_{sys.argv[1]}.json'))
    else:
        return load_json(os.path.join('conf', 'exps', f'exp_{exp_n}.json'))
