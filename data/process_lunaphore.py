print('running script')
import os
import gc
import h5py
import json
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import rescale, resize
from skimage.exposure import rescale_intensity
import tifffile, dask.array as da
from process_cedar_biolib_immune import extract_cells
from lunaphore_channel_info import get_channel_info

def get_mask(fname):
    return imread(fname)
       
def get_image(fname):
    imdata = tifffile.imread(fname)
    return da.from_array(imdata, chunks=(1, 10000, 10000))

def norm_channel(ch):
    print(ch.shape, ch.min(), ch.max())
    ch = np.log(ch + 1)
    print(ch.shape, ch.min(), ch.max())
    ch = rescale_intensity(ch, in_range=(np.percentile(ch[ch>0], 0.1), np.percentile(ch[ch>0], 99.9)), out_range='uint8')
    return ch


def norm_if(IF):
    # Normalize each channel
    normalized_channels = [norm_channel(IF[i].compute()) for i in range(IF.shape[0])]
    # Apply normalization to each channel
    return np.stack(normalized_channels)

if __name__ == '__main__':
    print('entering main')
    CROP_SIZE = 64
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    
    save_dir =  '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm'
    if not os.path.exists(save_dir): os.mkdir(save_dir)  

    with open('lunaphore_sample_info.json') as f:
        samples = json.load(f)
    
    for sample_name in list(samples.keys()):
        if not (os.path.exists(samples[sample_name]['IF']) and os.path.exists(samples[sample_name]['cell-mask'])):
            continue
        print(sample_name)
        save_fname = f'lunaphore_dataset_unnorm_sid={sample_name}.h5'
            
        print('retrieving samples...')
        IF = get_image(samples[sample_name]['IF'])
        orig_shape = IF.shape
        
        def rescale_channel(channel):
            return rescale(channel, 0.8, anti_aliasing=False, preserve_range=True)
        IF = da.map_blocks(rescale_channel, IF)
        
        print('retrieving cell mask...')
        cell_mask = get_mask(samples[sample_name]['cell-mask'])
        cell_mask = resize(cell_mask, orig_shape[1:], order=0)
        cell_mask = rescale(cell_mask, 0.8, order=0)
        print(cell_mask.shape)
    
        print('normalizing IF...')
        IF = norm_if(IF)
        
        print('padding images...')
        pad = ((0,),(CROP_SIZE,), (CROP_SIZE,))
        IF, cell_mask = np.pad(IF, pad), np.pad(cell_mask, CROP_SIZE)
    
        print(f'extracting cells from wsi...')
        masks, images, metadata = extract_cells(IF, cell_mask, sample_name, save_dir, CROP_SIZE, len(keep_channels))
        
        print(f'saving cells to hdf5 file...')
        with h5py.File(f'{save_dir}/{save_fname}', 'w') as f:
            images = f.create_dataset('images',data=np.stack(images))
            masks = f.create_dataset('masks',data=np.stack(masks))
            metas = f.create_dataset('metadata',data=metadata)
            