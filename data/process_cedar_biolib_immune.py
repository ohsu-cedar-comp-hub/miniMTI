print('importing libraries')
import os
import math
import gc
import re
from tqdm import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt
from einops import repeat, rearrange
from skimage.util import img_as_uint
from skimage.io import imread, imshow, imsave
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, resize
from cell_transformations import flip_mask, rotate_image
from PIL import Image
import zarr, tifffile, dask.array as da
Image.MAX_IMAGE_PIXELS = 1000000000   
np.seterr(divide='ignore')

def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    RoundsCyclesTable = 'RoundsCyclesTable.txt'
    try:
        assert os.path.exists('RoundsCyclesTable.txt')
    except AssertionError:
        RoundsCyclesTable = '../data/RoundsCyclesTable.txt'
        
    with open(RoundsCyclesTable) as f:
        channels = []
        for l in f.readlines():
            ch_name = l.split(' ')[0]
            if l.split(' ')[2] == 'c2': channels.extend([f"DAPI_{l.split(' ')[1]}", ch_name])
            else: channels.append(ch_name)

    keep_channels = [ch for ch in channels if (('DAPI' not in ch) or (('DAPI' in ch) and (ch == 'DAPI_R1'))) and (ch != 'R4c2')]
    reduced_set = ['DAPI_R1','aSMA','AMACR','CD163','CD66b','CD68','CD20','CD3','CD45','FOXP3','CD11b','CD4','PD1',
               'HLADRB1','Tryp','Ki67','Ecad','CK5','CK8','ChromA','Vim','GZMB','AR','NKX31','NCAM',
               'CD44','CD90','H3K27ac']
    keep_channels = [ch for ch in channels if ch in reduced_set]
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx
    
#normalize IF channel by clipping intensity range and converting to uint8
def norm_if_channel(ch):
    ch = np.log(ch)
    ch[ch < 0] = 0
    #ch = np.clip(ch, np.percentile(ch[ch>0], 1), np.percentile(ch[ch>0], 99))
    #ch = rescale_intensity(ch, in_range=(0, np.log(2**16)), out_range='uint8')
    ch = rescale_intensity(ch, in_range=(np.percentile(ch[ch>0], 0.1), np.percentile(ch[ch>0], 99.9)), out_range='uint8')
    return ch

#normalize all IF channels
def norm_if(IF):
    output = np.empty(IF.shape, dtype='uint8')
    for i,ch in tqdm(enumerate(IF)):
        output[i] = norm_if_channel(ch)
    return output

#normalize all IF channels
def norm_if_cm(IF, IF_cm):
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    cm_markers = ['DAPI_R1','aSMA', 'HLADRB1','Ecad', 'Vim', 'CK8','H3K27ac']
    output = np.empty(IF.shape, dtype='uint8')
    for i,ch in tqdm(enumerate(IF)):
        if keep_channels[i] in cm_markers:
            output[i] = norm_if_channel(IF_cm[cm_markers.index(keep_channels[i])])
        else:
            output[i] = norm_if_channel(ch)
    return output

def get_mask(sample_name):
    mask_dir = '/home/groups/ChangLab/dataset/biolib-immune/biolib-immune-cell-masks-last-round'
    fname = [f for f in os.listdir(mask_dir) if (sample_name in f) and ('cell_mask' in f)][0]
    return imread(f'{mask_dir}/{fname}')
    #return rescale(imread(f'{mask_dir}/{fname}'), 0.5, order=0)
    
    
def get_image(sid):
    #image_dir = '/home/groups/ChangLab/dataset/biolib-immune/images'
    #fname = [f for f in os.listdir(image_dir) if sid in f][0]
    #fname = f'{image_dir}/{fname}'
    fname = f'/home/groups/CEDAR/panel-reduction-project/biolib-immune-normalized-ometiffs/{sid}/{sid}_normalized.ome.tiff'
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)
    #return np.stack([img_as_uint(rescale(ch, 0.5)) for ch in im])
    
def get_cm_image(sid):
    '''image normed from cell mask'''
    #image_dir = '/home/groups/ChangLab/dataset/biolib-immune/images'
    #fname = [f for f in os.listdir(image_dir) if sid in f][0]
    #fname = f'{image_dir}/{fname}'
    fname = f'/home/groups/CEDAR/panel-reduction-project/biolib-immune-normalized-ometiffs-reduced-reduced/{sid}/{sid}_normalized.ome.tiff'
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)
    #return np.stack([img_as_uint(rescale(ch, 0.5)) for ch in im])


def extract_cells(IF, cell_mask, sample_name, save_dir, crop_size, num_channels):
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
        mask = repeat(mask, 'h w -> c h w', c=num_channels)

        #check for segmentation error
        if im[0].mean() <= 125: 
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
    print('entering main')
    CROP_SIZE = 64
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    
    save_dir =  '/home/groups/CEDAR/panel-reduction-project/train-data/biolib-immune-norm'
    if not os.path.exists(save_dir): os.mkdir(save_dir)  

    #ids = ['17633-6', '18538-6', '24952-6','30411-6', '31022-6', '31480-6', '33548-6', '38592-6', '48411-6', '54774-4', '57494-6', '57658-6', '19142-6']
    ids = ['19142-6']
    
    for sample_name in ids:
        save_fname = f'biolib_immune_dataset_normed_sid={sample_name}.h5'
            
        print('retrieving samples...')
        IF = get_image(sample_name)
        IF_cm = get_cm_image(sample_name)
        IF_cm = IF_cm.compute()
        #IF = IF[keep_channels_idx].compute()
        IF = IF.compute()
        print(IF.shape)
        
        print('retrieving cell mask...')
        cell_mask = get_mask(sample_name)
        print(cell_mask.shape)
    
        print('normalizing IF...')
        #IF = norm_if(IF)
        IF = norm_if_cm(IF, IF_cm)
    
        print('padding images...')
        pad = ((0,),(CROP_SIZE,), (CROP_SIZE,))
        IF, cell_mask = np.pad(IF, pad), np.pad(cell_mask, CROP_SIZE)
        print(IF.shape)
    
        print(f'extracting cells from wsi...')
        masks, images, metadata = extract_cells(IF, cell_mask, sample_name, save_dir, CROP_SIZE, len(keep_channels))
        
        print(f'saving cells to hdf5 file...')
        with h5py.File(f'{save_dir}/{save_fname}', 'w') as f:
            images = f.create_dataset('images',data=np.stack(images))
            masks = f.create_dataset('masks',data=np.stack(masks))
            metas = f.create_dataset('metadata',data=metadata)
        
        
     



