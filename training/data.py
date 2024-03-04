import os
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import rgb2gray
import histomicstk as htk
from einops import repeat, rearrange
from torch.utils.data import DataLoader
import numpy as np
import random

class SingleCellDataset(Dataset):
    def __init__(self, files, mask_dir, include_he, remove_background, grayscale, deconvolve):
        self.img_files = files
        self.mask_dir = mask_dir
        self.include_he = include_he
        self.remove_background = remove_background
        self.grayscale = grayscale
        self.deconvolve = deconvolve
        

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filepath = self.img_files[idx]
        im = imread(filepath)
        # whether to include H&E information
        if self.include_he == False: 
            im = im[:,:,:-3] #remove last three channels (H&E RGB channels)
        #else:
        elif self.grayscale == True: #replace DAPI with grayscale H&E
            he_im = im[:,:,-3:].copy() #slice off H&E channels
            he_im = rgb2gray(he_im) #convert H&E to grayscale
            im = np.concatenate([np.expand_dims(he_im, 2),im[:,:,1:-3]], axis=2) #restack IF channels(minus DAPI) and grayscale H&E channel
        elif self.deconvolve == True:
            stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
            # specify stains of input image
            stains = ['hematoxylin',  # nuclei stain
                      'eosin',        # cytoplasm stain
                      'null']         # set to null if input contains only two stains
            # create stain matrix
            W = np.array([stain_color_map[st] for st in stains]).T
            he_im = im[:,:,-3:].copy()
            imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(he_im, W)
            he_im = imDeconvolved.Stains[:,:,:2]
            im = np.concatenate([im[:,:,:-3], he_im], axis=2) #restack IF channels and deconvolved H and E channels
            
            
            
        num_channels = im.shape[-1]
        tensor = torch.from_numpy(im)
        tensor = rearrange(tensor, 'h w c -> c h w') 
        tensor = tensor.float()

        mask_path = os.path.join(self.mask_dir, os.path.basename(filepath).replace('.tif', '-mask.tif'))
        mask = imread(mask_path)
        mask = torch.tensor(mask.astype('bool'))
        
        if self.remove_background:
            if self.include_he: #only remove background for IF channels
                if self.grayscale:
                    mask = repeat(mask, 'h w -> c h w', c=num_channels - 1)
                    if_tensor = tensor[1:,:,:].clone()
                    if_tensor[mask == 0] = 0
                    tensor[1:,:,:] = if_tensor
                elif self.deconvolve:
                    mask = repeat(mask, 'h w -> c h w', c=num_channels - 2)
                    if_tensor = tensor[:-2,:,:].clone()
                    if_tensor[mask == 0] = 0
                    tensor[:-2,:,:] = if_tensor
                    
                else:
                    mask = repeat(mask, 'h w -> c h w', c=num_channels - 3)
                    if_tensor = tensor[:-3,:,:].clone()
                    if_tensor[mask == 0] = 0
                    tensor[:-3,:,:] = if_tensor
                            
            else:
                mask = repeat(mask, 'h w -> c h w', c=num_channels)
                tensor[mask == 0] = 0
            mask = mask[0]
            
        return tensor, mask, filepath

def get_train_dataloaders(batch_size, num_val_samples, include_he, remove_background, grayscale, deconvolve):
    data_dir = '/home/exacloud/gscratch/CEDAR/cycif-panel-reduction/biolib-immune-rescale'
    mask_dir = '/home/exacloud/gscratch/CEDAR/cycif-panel-reduction/biolib-immune-cell-masks-rescale'
    train_files = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if ('31480-6' not in f) and ('54774-4' not in f)]
    print(len(train_files))
    val_files = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if ('31480-6' in f)]
    random.Random(4).shuffle(train_files)
    random.Random(4).shuffle(val_files)
    train_data = SingleCellDataset(train_files[:-num_val_samples], mask_dir, include_he, remove_background, grayscale, deconvolve)
    val_data = SingleCellDataset(val_files[:num_val_samples], mask_dir, include_he, remove_background, grayscale, deconvolve)
    val2_data = SingleCellDataset(train_files[-num_val_samples:], mask_dir, include_he, remove_background, grayscale, deconvolve)

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
    val2_loader = DataLoader(val2_data, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         num_workers=1,
                         persistent_workers=True,
                         pin_memory=True)

    return train_loader, val_loader, val2_loader


def get_panel_selection_data(data_dir, val_dataset, mask_dir, batch_size, dataset_size, shuffle_data, include_he, remove_background, grayscale, deconvolve):

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
        print("-- Scanning files in ", val_dataset)
        files = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if any(ds in f for ds in val_dataset)]
        print("-- Randomly shuffling data")
        # TODO: make random shuffle a flag

        if shuffle_data: 
            random.seed(123)  # Set the seed for reproducibility
            random.shuffle(files)

        # Calculate the number of files and the dataset size
        num_files = len(files)

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
        val_files = files[:dataset_size]

        # for now, use False for remove_background
        print("-- Loading the dataset into dataloader")
        val_data = SingleCellDataset(val_files, mask_dir, include_he, remove_background, grayscale, deconvolve)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True)
        return val_loader
        
    except Exception as e:
        print("An error occurred while loading the testing dataset:", e)
        sys.exit(1)  # Exit the program with a non-zero exit code to indicate an error
