from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot
import torch
import numpy as np
import os
from conf import paths, general
import random
from torchvision.transforms.functional import hflip, vflip
from skimage.util import view_as_windows, view_as_blocks

class TreeTrainDataSet(Dataset):
    def __init__(self, path_to_patches, device, data_aug = False, transformer = ToTensor(), lidar_bands = None) -> None:
        opt_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'))
        lidar_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LIDAR}_img.npy'))
        if lidar_bands is not None:
            lidar_img = lidar_img[:, :, lidar_bands]

        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))
        self.lidar_img = lidar_img.reshape((-1, lidar_img.shape[-1]))

        #self.labels = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).reshape((-1,1)).astype(np.int64)
        self.labels = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).flatten().astype(np.int64)
        self.n_classes = np.unique(self.labels).shape[0]
        self.patches = np.load(path_to_patches)#[:200]
        self.transformer = transformer
        self.data_aug = data_aug

        self.device = device

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        patch_idx = self.patches[index]
        opt_tensor = self.transformer(self.opt_img[patch_idx]).to(self.device)
        lidar_tensor = self.transformer(self.lidar_img[patch_idx]).to(self.device)
        #label_tensor = self.transformer(self.labels[patch_idx].astype(np.int64)).squeeze(0).to(self.device)
        label_tensor = torch.tensor(self.labels[patch_idx]).to(self.device)

        if self.data_aug:
            k = random.randint(0, 3)
            opt_tensor = torch.rot90(opt_tensor, k, (1,2))
            lidar_tensor = torch.rot90(lidar_tensor, k, (1,2))
            label_tensor = torch.rot90(label_tensor, k, (0,1))

            if bool(random.getrandbits(1)):
                opt_tensor = hflip(opt_tensor)
                lidar_tensor = hflip(lidar_tensor)
                label_tensor = hflip(label_tensor)

            if bool(random.getrandbits(1)):
                opt_tensor = vflip(opt_tensor)
                lidar_tensor = vflip(lidar_tensor)
                label_tensor = vflip(label_tensor)

        return (
            (
                opt_tensor,
                lidar_tensor
            ),
            label_tensor
        )


class TreePredDataSet(Dataset):
    def __init__(self, device, overlap = 0, transformer = ToTensor(), lidar_bands = None) -> None:
        opt_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'))
        lidar_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LIDAR}_img.npy'))

        if lidar_bands is not None:
            lidar_img = lidar_img[:, :, lidar_bands]

        self.transformer = transformer
        self.device = device
        self.original_shape = opt_img.shape[:2]

        pad_shape = ((general.PATCH_SIZE, general.PATCH_SIZE),(general.PATCH_SIZE, general.PATCH_SIZE),(0,0))

        opt_img = np.pad(opt_img, pad_shape, mode = 'reflect')
        lidar_img = np.pad(lidar_img, pad_shape, mode = 'reflect')
        shape = opt_img.shape[:2]

        window_step = int(general.PATCH_SIZE*(1-overlap))

        idx = np.arange(shape[0]*shape[1]).reshape(shape)
        self.padded_shape = shape

        self.idx_patches = view_as_windows(idx, (general.PATCH_SIZE, general.PATCH_SIZE), window_step).reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))

        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))
        self.lidar_img = lidar_img.reshape((-1, lidar_img.shape[-1]))

    def __len__(self):
        return self.idx_patches.shape[0]

    def __getitem__(self, index):
        patch_idx = self.idx_patches[index]
        opt_tensor = self.transformer(self.opt_img[patch_idx]).to(self.device)
        lidar_tensor = self.transformer(self.lidar_img[patch_idx]).to(self.device)

        return (
                opt_tensor,
                lidar_tensor
            )
