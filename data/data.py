import os
import sys
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.transform import rescale
from einops import repeat, rearrange


class SingleCellDataset(Dataset):
    def __init__(self, images, masks, metadata, downscale=False, remove_background=False, remove_he=True, deconvolve_he=True, rescale=True, no_he=False, train=False):
        self.images = images
        self.masks = masks
        self.metadata = metadata
        self.remove_background = remove_background
        self.remove_he = remove_he
        self.downscale = downscale
        self.deconvolve_he = deconvolve_he
        self.rescale = rescale
        self.no_he = no_he
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not self.train:
            meta = self.metadata[idx]

        # Add retry logic for HDF5 I/O errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                im = self.images[idx]
                mask = self.masks[idx]
                break
            except OSError as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    raise e

        if self.downscale:
            im = rescale(im, 0.5, channel_axis=2, preserve_range=True)
            mask = rescale(mask, 0.5, order=0)

        tensor = torch.from_numpy(im)
        tensor = rearrange(tensor, 'h w c -> c h w')

        if self.remove_he:
            tensor = tensor[:-3]
        else:
            if self.deconvolve_he and not self.no_he:
                he = tensor[-3:].numpy()
                he = np.moveaxis(he, 0, 2)
                import histomicstk as htk
                stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
                stains = ['hematoxylin', 'eosin', 'null']
                W = np.array([stain_color_map[st] for st in stains]).T
                imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(he, W)
                he = imDeconvolved.Stains[:, :, :2]
                tensor = np.concatenate([tensor[:-3], torch.from_numpy(np.moveaxis(he, 2, 0))], axis=0)
                tensor = torch.from_numpy(tensor)

        tensor = tensor.float()
        num_channels = tensor.shape[0]

        mask = torch.tensor(mask.astype('bool'))

        if self.remove_background:
            mask = repeat(mask, 'h w -> c h w', c=num_channels)
            tensor[mask == 0] = 0
            mask = mask[0]

        if self.rescale:
            if self.remove_he or self.no_he:
                tensor = (tensor / 127.5) - 1.0
            else:
                tensor = (tensor / 127.5) - 1.0

        if self.train:
            return tensor, mask
        return tensor, mask, meta


def get_train_dataloaders(train_file, val_file, batch_size, num_val_samples, downscale=False, remove_background=False, remove_he=True, deconvolve_he=True, rescale=True):
    train_file = h5py.File(train_file)
    val_file = h5py.File(val_file)

    train_data = SingleCellDataset(train_file['images'], train_file['masks'], train_file['metadata'], downscale, remove_background, remove_he, deconvolve_he, rescale, train=True)
    val_data = SingleCellDataset(val_file['images'], val_file['masks'], val_file['metadata'], downscale, remove_background, remove_he, deconvolve_he, rescale, train=True)

    train_loader = DataLoader(train_data,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=32,
                       persistent_workers=True,
                       pin_memory=True)
    val_loader = DataLoader(val_data,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=16,
                         persistent_workers=True,
                         pin_memory=True)

    return train_loader, val_loader


def get_panel_selection_data(val_file, batch_size, dataset_size, remove_background=False, downscale=False, remove_he=True, deconvolve_he=True, shuffle=True, rescale=True):
    """
    Retrieves a DataLoader for validation datasets.

    Parameters:
        val_file: Path(s) to validation HDF5 file(s).
        batch_size (int): Batch size for the DataLoader.
        dataset_size: Number of samples to use, or 'all'/'All' for the full dataset.
        remove_background (bool): Whether to zero out background pixels.
        downscale (bool): Whether to downscale images by 2x.
        remove_he (bool): Whether to remove H&E channels.
        deconvolve_he (bool): Whether to deconvolve H&E stains.
        shuffle (bool): Whether to shuffle indices.
        rescale (bool): Whether to rescale pixel values to [-1, 1].

    Returns:
        DataLoader: A DataLoader containing the validation data.
    """

    try:
        val_file = h5py.File(val_file[0])
        random.seed(123)
        idx = np.arange(len(val_file['images']))
        if shuffle:
            random.shuffle(idx)
        idx = sorted(idx)

        num_files = len(idx)

        if dataset_size in ['all', 'All']:
            print(f"-- Using all of the validation dataset, size={num_files}")
            dataset_size = num_files
        else:
            if type(dataset_size) is int and dataset_size > num_files:
                print(f"-- dataset_size larger than available files. Using all {num_files} files instead.")
                dataset_size = num_files
            elif type(dataset_size) is int and dataset_size <= 0:
                raise ValueError("The calculated dataset_size is less than or equal to 0, which is invalid.")

        dataset_percentage = (dataset_size / num_files) * 100
        print(f"-- Using {dataset_percentage:.1f}% of the dataset, selecting {dataset_size} files out of {num_files} files")
        idx = idx[:dataset_size]

        print("-- Loading the dataset into dataloader")
        val_data = SingleCellDataset(val_file['images'][idx], val_file['masks'][idx], val_file['metadata'][idx], downscale, remove_background, remove_he, deconvolve_he, rescale)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=10, persistent_workers=True, pin_memory=True)
        return val_loader

    except Exception as e:
        print("An error occurred while loading the testing dataset:", e)
        sys.exit(1)
