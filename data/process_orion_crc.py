import os
import h5py
import numpy as np
from typing import Union, List
from pathlib import Path
from skimage.io import imread
import zarr, tifffile, dask.array as da
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, resize
from tqdm import tqdm
from einops import repeat
from cell_transformations import flip_mask, rotate_image
import math
import gc
import histomicstk as htk
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.preprocessing.color_conversion import lab_mean_std

def tifffile_to_dask(im_fp: Union[str, Path]) -> Union[da.array, List[da.Array]]:
    imdata = zarr.open(tifffile.imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = [da.from_zarr(imdata[z], chunks=(19,10000,10000)) for z in imdata.array_keys()]
    else:
        imdata = da.from_zarr(imdata, chunks=(19,10000,10000))
    return imdata

def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    channels = [
        'DAPI',
        'AF1',
        'CD31',
        'CD45',
        'CD68',
        'Argo550',
        'CD4',
        'FOXP3',
        'CD8a',
        'CD45RO',
        'CD20',
        'PD-L1',
        'CD3e', #Cd3?
        'CD163',
        'E-cadherin',
        'PD-1',
        'Ki67',
        'PanCK',
        'aSMA']

    keep_channels = [ch for ch in channels if ch != "AF1" and ch != 'Argo550']
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx

def norm_if_channel(ch):
    ch = np.log(ch)
    ch[ch < 0] = 0
    #ch = rescale_intensity(ch, in_range=(np.percentile(ch[ch>0], 0.1), np.percentile(ch[ch>0], 99.9)), out_range='uint8')
    ch = rescale_intensity(ch, in_range=(np.percentile(ch, 5), np.percentile(ch, 99.9)), out_range='uint8')
    return ch

#normalize all IF channels
def norm_if(IF):
    output = np.empty(IF.shape, dtype='uint8')
    for i,ch in tqdm(enumerate(IF)):
        output[i] = norm_if_channel(ch)
    return output

print('loading reference HE')
#calculate target parameters as global variables
#load target image (e.g. CRC08)
#loading second pyramid level (0.65 mpp)
target_im_path = '/home/groups/ChangLab/dataset/Orion_CRC/CRC02/18459_LSP10364_US_SCAN_OR_001__092347-registered.ome.tif'
target = tifffile_to_dask(target_im_path)[1].compute()
if target.shape[0] == 3: ref = np.moveaxis(target, 0, 2)
#load tissue mask for target image
target_mask_path = "/home/groups/ChangLab/dataset/Orion_CRC/CRC08/CRC08_reinhard_mask.tiff"
target_mask = imread(target_mask_path)
#resize tissue mask (to match second pyramid level)
target_mask = resize(target_mask, target.shape[:-1], order=0)
#calculate mu and sigma for LAB-converted target image
mu, sigma = lab_mean_std(target, mask_out=target_mask)

#normalize H&E using reinhard normalization
def norm_he(ref, ref_mask):
    global mu, sigma
    #make sure image is in HxWxC format
    if ref.shape[0] == 3: ref = np.moveaxis(ref, 0, 2)
    #load and resize tissue mask
    ref_mask = resize(ref_mask, ref.shape[:-1], order=0)
    #return normalized image
    normed_ref = reinhard(ref, mu, sigma, mask_out=ref_mask)
    #return back to CxHxW
    normed_ref = np.moveaxis(normed_ref, 2, 0)
    assert normed_ref.shape[0] == 3, 'HE not in CxHxW'
        
    return normed_ref

def get_IF(sid):
    fname = f'/home/groups/ChangLab/wangmar/shift-panel-reduction/shift-panel-reduction-main/landmark_normalization/cycif-normalization-pipeline/command_line_script/final_script/new_normalized_crc_dataset/{sid}/{sid}_normalized.ome.tiff'
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)


#def get_HE(sid):
#    fname = f'/home/groups/ChangLab/dataset/Orion_CRC/{sid}/{sid}_normalized_HE_2nd-res_deconv.tiff'
#    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
#    return da.from_zarr(imdata)

def get_HE(sample_name):
    data_dir = '/home/groups/ChangLab/dataset/Orion_CRC/'
    print(f'loading {sample_name=}')
    if sample_name == 'CRC01': 
        HE_path = '/home/groups/ChangLab/dataset/Orion_CRC/CRC01/registration/18459-LSP10353-US-SCAN-OR-001 _093059-registered.ome.tif'
        HE_tissue_mask_path = '/home/groups/ChangLab/dataset/Orion_CRC/CRC01/CRC01_reinhard_mask.tiff'
    else:
        sample_dir = f'{data_dir}/{sample_name}'
        HE_fname = [f for f in os.listdir(sample_dir) if f.endswith('-registered.ome.tif')][0]
        HE_tissue_mask_path = f'{data_dir}/{sample_name}/{sample_name}_reinhard_mask.tiff'
        HE_path = f'{sample_dir}/{HE_fname}'
        
    if sample_name == 'CRC01':
        HE = tifffile_to_dask(HE_path)[0].compute()
        HE = rescale(HE, 0.5, channel_axis=0)
    else:
        HE = tifffile_to_dask(HE_path)[1].compute()
    HE_tissue_mask = imread(HE_tissue_mask_path)
    
    return HE, HE_tissue_mask

def get_mask(sid):
    mask_fname = f'/home/groups/ChangLab/dataset/Orion_CRC/mesmer_masks/{sid.lower()}_mesmer_cell_mask.tif'
    return imread(mask_fname)
 
def get_samples(data_dir, sample_name):
    '''get unnormalized samples'''
    print(f'loading {sample_name=}')
    if sample_name == 'CRC01': 
        IF_path = '/home/groups/ChangLab/dataset/Orion_CRC/CRC01/registration/P37_S29_A24_C59kX_E15__at__20220106_014304_946511.ome.tiff'
        HE_path = '/home/groups/ChangLab/dataset/Orion_CRC/CRC01/registration/18459-LSP10353-US-SCAN-OR-001 _093059-registered.ome.tif'
        HE_tissue_mask_path = '/home/groups/ChangLab/dataset/Orion_CRC/CRC01/CRC01_reinhard_mask.tiff'
    else:
        sample_dir = f'{data_dir}/{sample_name}'
        IF_fname = [f for f in os.listdir(sample_dir) if f.startswith('P37') and f.endswith('-zlib.ome.tiff')][0]
        HE_fname = [f for f in os.listdir(sample_dir) if f.endswith('-registered.ome.tif')][0]
        HE_tissue_mask_path = f'{data_dir}/{sample_name}/{sample_name}_reinhard_mask.tiff'
        IF_path = f'{sample_dir}/{IF_fname}'
        HE_path = f'{sample_dir}/{HE_fname}'
        
    IF = tifffile_to_dask(IF_path)[1]
    if sample_name == 'CRC01':
        HE = tifffile_to_dask(HE_path)[0].compute()
        HE = rescale(HE, 0.5, channel_axis=0)
    else:
        HE = tifffile_to_dask(HE_path)[1].compute()
    HE_tissue_mask = imread(HE_tissue_mask_path)
    
    return IF, HE, HE_tissue_mask

def extract_cells(IF, HE, cell_mask, sample_name, save_dir, crop_size, num_channels):
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
        if_im = IF[:, xmin:xmax, ymin:ymax].copy()
        he_im = HE[:, xmin:xmax, ymin:ymax].copy()
        mask = cell_mask[xmin:xmax, ymin:ymax].copy()

        #isolate single cell
        mask[mask != rp.label] = 0
        #binarize mask
        mask[mask > 0] = 1

        mask = mask.astype('uint8')
        #repeat mask for all channels
        mask = repeat(mask, 'h w -> c h w', c=num_channels)
        
        #move channel dimension
        he_im = np.moveaxis(he_im, 0, 2)
        if_im = np.moveaxis(if_im, 0, 2)
        mask = np.moveaxis(mask, 0, 2)
        
        '''
        #deconvolve
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        # specify stains of input image
        stains = ['hematoxylin',  # nuclei stain
                  'eosin',        # cytoplasm stain
                  'null']         # set to null if input contains only two stains
        # create stain matrix
        W = np.array([stain_color_map[st] for st in stains]).T
        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(he_im, W)
        he_im = imDeconvolved.Stains[:,:,:2]
        '''

        im = np.concatenate([if_im, he_im], axis=-1)
        
        #check for segmentation error
        if im[0].mean() == 0: 
            num_removed_from_seg += 1
            continue           
        

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

    save_dir =  '/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized'
    if not os.path.exists(save_dir): os.mkdir(save_dir)  
    sample_ids = ['01']
    
    for sample_id in sample_ids:
        sample_id = f'CRC{sample_id}'
        save_fname = f'orion_crc_dataset_sid={sample_id}.h5'
        print('retrieving samples...')
        IF, HE, HE_tissue_mask = get_samples(data_dir, sample_name)
        #IF = get_IF(sample_id)
        IF = IF[keep_channels_idx].compute()
        #HE, HE_mask = get_HE(sample_id)
        if HE.shape[-1] == 3:
            HE = np.moveaxis(HE,2,0)
        if sample_id != 'CRC02':
            print('normalizing H&E...')
            HE = norm_he(HE, HE_mask)

        cell_mask = get_mask(sample_id)
        print(IF.shape, HE.shape, cell_mask.shape)

        print('normalizing IF...')
        IF = norm_if(IF)

        print('padding images...')
        pad = ((0,),(CROP_SIZE,), (CROP_SIZE,)) #pad 2nd and 3rd dimensions with CROP_SIZE pixels
        IF, HE, cell_mask = np.pad(IF, pad), np.pad(HE, pad), np.pad(cell_mask, CROP_SIZE)
        print(IF.shape, HE.shape)

        print(f'extracting cells from wsi...')
        masks, images, metadata = extract_cells(IF, HE, cell_mask, sample_id, save_dir, CROP_SIZE, len(keep_channels) + 3)
        
        print(f'saving cells to hdf5 file...')
        with h5py.File(f'{save_dir}/{save_fname}', 'w') as f:
            images = f.create_dataset('images',data=np.stack(images))
            masks = f.create_dataset('masks',data=np.stack(masks))
            metas = f.create_dataset('metadata',data=metadata)
            
            