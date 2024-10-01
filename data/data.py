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
    def __init__(self, images, masks, metadata, downscale=False, remove_background=False, remove_he=True, deconvolve_he=False):
        self.images = images
        self.masks = masks
        self.metadata = metadata
        self.remove_background =remove_background
        self.remove_he = remove_he
        self.downscale = downscale
        self.deconvolve_he = deconvolve_he

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        im = self.images[idx]
        mask = self.masks[idx]
        if self.downscale:
            im = rescale(im, 0.5, channel_axis=2, preserve_range=True)
            mask = rescale(mask, 0.5, order=0)
        tensor = torch.from_numpy(im)
        tensor = rearrange(tensor, 'h w c -> c h w') 
        if self.remove_he:
            tensor = tensor[:-3] #cut off the last three channels (which are the H&E channels)
        else:
            if self.deconvolve_he:
                he = tensor[-3:]
                import histomicstk as htk
                stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
                # specify stains of input image
                stains = ['hematoxylin',  # nuclei stain
                          'eosin',        # cytoplasm stain
                          'null']         # set to null if input contains only two stains
                W = np.array([stain_color_map[st] for st in stains]).T
                imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(he, W)
                he = imDeconvolved.Stains[:,:,:2]
                tensor = np.concatenate([tensor[:-3], he], axis=0) #restack IF channels and deconvolved H and E channels
            
        tensor = tensor.float()
        num_channels = tensor.shape[0]

        mask = torch.tensor(mask.astype('bool'))
        
        if self.remove_background:
            mask = repeat(mask, 'h w -> c h w', c=num_channels)
            tensor[mask == 0] = 0
            mask = mask[0]
        tensor = (tensor / 127.5) - 1.0   
        return tensor, mask, meta

    
def get_train_dataloaders(train_file, val_file, batch_size, num_val_samples, downscale=False, remove_background=False, remove_he=True, deconvolve_he=False):
    train_file = h5py.File(train_file)
    val_file = h5py.File(val_file)
    
    train_data = SingleCellDataset(train_file['images'], train_file['masks'], train_file['metadata'], downscale, remove_background, remove_he, deconvolve_he)
    val_data = SingleCellDataset(val_file['images'], val_file['masks'], val_file['metadata'], downscale, remove_background, remove_he, deconvolve_he)

    train_loader = DataLoader(train_data, 
                       batch_size=batch_size, 
                       shuffle=True, 
                       num_workers=1,
                       persistent_workers=True,
                       pin_memory=True)
    val_loader = DataLoader(val_data, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         num_workers=1,
                         persistent_workers=True,
                         pin_memory=True)
    
    return train_loader, val_loader 


def get_panel_selection_data(val_file, batch_size, dataset_size, remove_background=False, downscale=False, remove_he=True, deconvolve_he=False, shuffle=True):

    """
    Retrieves a DataLoader for validation datasets in CRC dataset.

    This function filters and prepares a DataLoader object for a given list of validation datasets. 
    It scans a specified directory for files that match any dataset names in the provided list, 
    randomly shuffles these files, and selects a subset of them for the DataLoader. 
    The DataLoader is then created using the SingleCellDataset class, considering additional 
    parameters for inclusion of HE staining and background removal options.

    Parameters:
    data_dir (str): path to the directory where data is saved
    val_dataset (list of str): A list of dataset names to include in the validation data.
    BATCH_SIZE (int): The size of each batch to be loaded by the DataLoader.
    include_he (bool): A flag to determine whether to include HE staining data.
    remove_background (bool): A flag to determine whether to remove background in the data preprocessing.
    dataset_size (int): size of the validation dataset to use for inferencing

    Returns:
    DataLoader: A DataLoader object containing the validation data.

    Example:
    >>> val_loader = get_panel_selection_data(["CRC05", "CRC06"], 10000, True, False)
    """
    

    try: 
        val_file = h5py.File(val_file[0])
        random.seed(123)  # Set the seed for reproducibility
        idx = np.arange(len(val_file['images']))
        if shuffle:
            random.shuffle(idx)
        idx = sorted(idx)

        # Calculate the number of files and the dataset size
        num_files = len(idx)

        if dataset_size in ['all', 'All']:
            print(f"-- Using all of the validation dataset, size={num_files}")
            dataset_size = num_files

        else: 
            if type(dataset_size) is int and dataset_size > num_files:
                print(f"-- dataset_percentage results in a dataset_size larger than available files. Using all {num_files} files instead.")
                dataset_size = num_files
            elif type(dataset_size) is int and dataset_size <= 0:
                raise ValueError("The calculated dataset_size is less than or equal to 0, which is invalid.")

        dataset_percentage = (dataset_size/num_files) * 100

        print(f"-- Using {dataset_percentage}% of the dataset, selecting {dataset_size} files out of {num_files} files")
        idx = idx[:dataset_size]

        # for now, use False for remove_background
        print("-- Loading the dataset into dataloader")
        val_data = SingleCellDataset(val_file['images'][idx], val_file['masks'][idx], val_file['metadata'][idx], downscale, remove_background, remove_he, deconvolve_he)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True)
        return val_loader
        
    except Exception as e:
        print("An error occurred while loading the testing dataset:", e)
        sys.exit(1)  # Exit the program with a non-zero exit code to indicate an error
