import os
import math
import gc
import re
from tqdm import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt
from einops import repeat, rearrange
from skimage.io import imread, imshow, imsave
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, resize
from cell_transformations import flip_mask, rotate_image
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000   

np.seterr(divide='ignore')

def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    RoundsCyclesTable = 'RoundsCyclesTable.txt'
    with open(RoundsCyclesTable) as f:
        channels = []
        for l in f.readlines():
            ch_name = l.split(' ')[0]
            if l.split(' ')[2] == 'c2': channels.extend([f"DAPI_{l.split(' ')[1]}", ch_name])
            else: channels.append(ch_name)

    keep_channels = [ch for ch in channels if ('DAPI' not in ch) or (('DAPI' in ch) and (ch == 'DAPI_R1'))]
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx
    
#normalize IF channel by clipping intensity range and converting to uint8
def norm_if_channel(ch):
    ch = np.log(ch)
    ch[ch == -np.inf] = 0
    clip_range = (np.percentile(ch, 0.1), np.percentile(ch, 99.9))
    ch = rescale_intensity(ch, in_range=clip_range, out_range='uint8')
    return ch

#normalize all IF channels
def norm_if(IF):
    increase_min = [i for i in range(IF.shape[0])]
    output = np.empty(IF.shape, dtype='uint8')
    for i,ch in tqdm(enumerate(IF)):
        output[i] = norm_if_channel(ch)
 
    return output


def get_sample(sample_dir, keep_channels_idx):
    IF_channels = [np.expand_dims(imread(f), 0) for i,f in enumerate(sample_dir) if i in keep_channels_idx]
    return np.concatenate(IF_channels, axis=0)


def extract_cells(IF, cell_mask, sample_name, save_dir, mask_save_dir, crop_size, num_channels):
    #iterate through cell regions
    rps = regionprops(cell_mask.astype('int'))
    num_removed_from_size = 0
    num_removed_from_seg = 0
    
    masks, images, metadata = [], [], [] 
    for rp in tqdm(rps):

        #size filter, make sure bbox for cell mask is not bigger than 32x32px
        min_row, min_col, max_row, max_col = rp.bbox
        if ((max_row - min_row) > crop_size) or ((max_col - min_col) > crop_size):
            num_removed_from_size += 1
            continue

        #get centroid of cell and and bbox by expanding out in each direction by 32 pixels to obtain 64x64px image
        #we will later crop down to 32x32px, but this is to ensure we have enough context when performing transformations
        center_x, center_y = int(rp.centroid[0]), int(rp.centroid[1])
        xmin, xmax, ymin, ymax = center_x - crop_size, center_x + crop_size, center_y - crop_size, center_y + crop_size

        #crop image and mask
        #im = wsi[:, xmin:xmax, ymin:ymax].compute().copy()
        im = IF[:, xmin:xmax, ymin:ymax].copy()
        mask = cell_mask[xmin:xmax, ymin:ymax].copy()

        #isolate single cell
        mask[mask != rp.label] = 0
        #binarize mask
        mask[mask > 0] = 1

        mask = mask.astype('uint8')
        #repeat mask for all channels
        mask = repeat(mask, 'h w -> c h w', c=len(keep_channels))

        #check for segmentation error
        if im[0].mean() == 0: 
            num_removed_from_seg += 1
            continue 
            
        #move channel dimension
        im = np.moveaxis(im, 0, 2)
        mask = np.moveaxis(mask, 0, 2)

        #perform transformations
        im = rotate_image(im,-math.degrees(rp.orientation))
        mask = rotate_image(mask, -math.degrees(rp.orientation))
        im, mask = flip_mask(im, mask, crop_size)

        #undo repeat op
        mask = mask[:,:,0]
        
        #crop to 32x32px
        im = im[int(crop_size/2):-int(crop_size/2), int(crop_size/2):-int(crop_size/2),:]
        mask = mask[int(crop_size/2):-int(crop_size/2), int(crop_size/2):-int(crop_size/2)]
        
        assert im.shape == (crop_size,crop_size,num_channels), f"error im not in HxWxC, {im.shape=}"
        assert mask.shape == (crop_size,crop_size), f"error, mask shape not 32x32, {mask.shape=}"
        
        meta = f'{sample_name}-CellID-{rp.label}-x={center_x}-y={center_y}'
        masks.append(mask)
        images.append(im)
        metadata.append(meta)

    print(f'finished processing sample {sample_name}, {num_removed_from_size=}, {num_removed_from_seg=}')
    del IF
    del cell_mask
    gc.collect()
    
    return masks, images, metadata
    

if __name__ == '__main__':  
    CROP_SIZE = 32
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    
    save_dir =  '/home/groups/ChangLab/dataset/cycif-panel-reduction/biolib-immune'
    save_fname = 'biolib_immune_dataset_rescaled.h5'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    if not os.path.exists(mask_save_dir): os.mkdir(mask_save_dir)
    
    biolib_immune_dir = '/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/RegisteredImages/'
    subdirs = os.listdir(biolib_immune_dir)
    id_query = r'[0-9]{5}-[0-9]{1}'
    ids = set()
    for dir in subdirs:
        if (x := re.search(id_query, dir)):
            ids.add(x[0])
            
    need_crop = ['31022-6', '33548-6', '17633-6', '18538-6']
    ids = [i for i in ids if i not in need_crop]
    sample_dirs = []
    for id_ in ids:
        subdirs = os.listdir(biolib_immune_dir)
        id_folder = [f for f in subdirs if f.startswith(f"{id_}")][0]
        sample_dirs.append(f'{biolib_immune_dir}/{id_folder}')
    
    max_round = 10
    max_channel = 5
    sorted_sample_dirs = []
    for dir_ in sample_dirs:
        sorted_files = []
        for round in range(1, max_round+1):
            for ch in range(1, max_channel+1):
                files = [f for f in os.listdir(dir_) if (f'R{round}_' in f) and (f'_c{ch}_' in f)]
                if len(files) > 0: sorted_files.append(f'{dir_}/{files[0]}')
        sorted_sample_dirs.append(sorted_files)
    
    mask_paths = [
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/19142-6_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.png",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/57658-6_AFSubtracted - Segmentation-do R10 dapi filter/Scene 001 - Cell Labels.tif",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/57494-6_AFSubtracted - Segmentation-ok/Scene 001 - Cell Labels.png",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/24952-6_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.tif",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/38592-6_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.png",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/48411-6_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.png",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/30411-6_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.tif",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/31480-6_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.tif",
"/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2020_Immune/SubtractedRegisteredImages/54774-4_AFSubtracted - Segmentation-corrected/Scene 001 - Cell Labels.tif"]
    masks, images, metadata = [], [], [] 
    for sample_dir, sample_name in zip(sorted_sample_dirs, ids):
        print('retrieving samples...')
        IF = get_sample(sample_dir, keep_channels_idx)
        IF = rescale(IF, 0.5, channel_axis=0)
        print(IF.shape)
        
        print('retrieving cell mask...')
        mask_path = [f for f in mask_paths if sample_name in f][0]
        cell_mask = imread(mask_path)
        cell_mask = rescale(cell_mask, 0.5, order=0)
        print(cell_mask.shape)
    
        print('normalizing IF...')
        IF = norm_if(IF)
    
        print('padding images...')
        pad = ((0,),(CROP_SIZE,), (CROP_SIZE,)) #pad 2nd and 3rd dimensions with 64 pixels
        IF, cell_mask = np.pad(IF, pad), np.pad(cell_mask, CROP_SIZE)
        print(IF.shape)
    
        print(f'extracting cells from wsi...')
        masks_, images_, metadata_, = extract_cells(IF, cell_mask, sample_name, save_dir, mask_save_dir, CROP_SIZE, len(keep_channels))
        masks.extend(masks_)
        images.extend(images_)
        metadata.extend(metadata_)
        
    with h5py.File(f'{save_dir}/{save_fname}', 'w') as f:
        images = f.create_dataset('images',data=np.stack(images))
        masks = f.create_dataset('masks',data=np.stack(masks))
        metas = f.create_dataset('cell-metadata',data=metadata)
        
     


