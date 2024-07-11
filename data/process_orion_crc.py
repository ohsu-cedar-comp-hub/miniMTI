import os
import h5py
import numpy as np
from skimage.io import imread
import zarr, tifffile, dask.array as da
from process_cedar_biolib_immune import norm_if
from skimage.measure import regionprops
from tqdm import tqdm
from einops import repeat
from cell_transformations import flip_mask, rotate_image
import math
import gc


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


def get_IF(sid):
    fname = f'/home/groups/ChangLab/wangmar/shift-panel-reduction/shift-panel-reduction-main/landmark_normalization/cycif-normalization-pipeline/command_line_script/final_script/new_normalized_crc_dataset/{sid}/{sid}_normalized.ome.tiff'
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)


def get_HE(sid):
    fname = f'/home/groups/ChangLab/dataset/Orion_CRC/{sid}/{sid}_normalized_HE_2nd-res_deconv.tiff'
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)


def get_mask(sid):
    mask_fname = f'/home/groups/ChangLab/dataset/Orion_CRC/mesmer_masks/{sid.lower()}_mesmer_cell_mask.tif'
    return imread(mask_fname)
 
    
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

        im = np.concatenate([if_im, he_im], axis=0)
        
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
    keep_channels, keep_channels_idx, ch2idx = get_channel_info('CRC')

    save_dir =  '/home/groups/ChangLab/dataset/ORION-CRC'
    if not os.path.exists(save_dir): os.mkdir(save_dir)  
    sample_ids = ['01','02','03','04','05','06']
    
    for sample_id in sample_ids:
        sample_id = f'CRC{sample_id}'
        save_fname = f'orion_crc_dataset_sid={sample_id}.h5'
        print('retrieving samples...')
        IF = get_IF(sample_id)
        IF = IF[keep_channels_idx].compute()
        HE = get_HE(sample_id)
        HE = HE.compute()
        HE = np.moveaxis(HE, 2, 0)
        cell_mask = get_mask(sample_id)
        print(IF.shape, HE.shape, cell_mask.shape)

        print('normalizing IF...')
        IF = norm_if(IF)

        print('padding images...')
        pad = ((0,),(CROP_SIZE,), (CROP_SIZE,)) #pad 2nd and 3rd dimensions with 64 pixels
        IF, HE, cell_mask = np.pad(IF, pad), np.pad(HE, pad), np.pad(cell_mask, CROP_SIZE)
        print(IF.shape, HE.shape)

        print(f'extracting cells from wsi...')
        masks, images, metadata = extract_cells(IF, HE, cell_mask, sample_id, save_dir, CROP_SIZE, len(keep_channels) + 2)
        
        print(f'saving cells to hdf5 file...')
        with h5py.File(f'{save_dir}/{save_fname}', 'w') as f:
            images = f.create_dataset('images',data=np.stack(images))
            masks = f.create_dataset('masks',data=np.stack(masks))
            metas = f.create_dataset('metadata',data=metadata)
            
            