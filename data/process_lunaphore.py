import os
import h5py
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
import zarr, tifffile, dask.array as da
from process_cedar_biolib_immune import norm_if, extract_cells
from lunaphore_channel_info import get_channel_info


def get_mask(sample_name):
    mask_dir = "/home/groups/CEDAR/panel-reduction-project/ACED-immune/mesmer_cell_mask_tissue_loss"
    fname = [f for f in os.listdir(mask_dir) if (sample_name in f) and ('cell_mask' in f)][0]
    return imread(f'{mask_dir}/{fname}')
    
    
def get_image(fname):
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)



if __name__ == '__main__':
    print('entering main')
    CROP_SIZE = 64
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    
    save_dir =  '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm'
    if not os.path.exists(save_dir): os.mkdir(save_dir)  

    with open('lunaphore_sample_info.json','r') as f:
        samples = json.load(f)
    
    for sample_name in samples.keys():
        save_fname = f'lunaphore_dataset_norm_sid={sample_name}.h5'
            
        print('retrieving samples...')
        IF = get_image(samples[sample_name]['IF'])
        IF = IF[keep_channels_idx].compute()
        IF = rescale(IF, 1.25, anti_aliasing=False, channel_axis=0)
        print(IF.shape)
        
        print('retrieving cell mask...')
        cell_mask = get_mask(samples[sample_name]['cell-mask'])
        print(cell_mask.shape)
    
        print('normalizing IF...')
        IF = norm_if(IF)
    
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
            