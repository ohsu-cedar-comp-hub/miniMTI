import os
import h5py
import numpy as np
from skimage.io import imread
import zarr, tifffile, dask.array as da
from process_cedar_biolib_immune import norm_if, extract_cells


def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    '''
    channels = ['DAPI1', 'c2', 'CD8', 'CD4', 'FOXP3', 
                'DAPI2', 'EPCAM', 'AR', 'CD11b', 'CD68', 
                'DAPI3', 'CD56', 'NCR1', 'CK8', 'CD45', 
                'DAPI4', 'aSMA', 'ERG', 'CD163', 'ChromA', 
                'DAPI5', 'CK14', 'Tryp', 'Ki67', 'CD66b', 
                'DAPI6', 'GZMB', 'ECAD', 'PD1', 'CK5', 
                'DAPI7', 'VIM', 'FOSB', 'CD31', 'Tbr2', 
                'DAPI8', 'CD45RA', 'NKX31', 'HLADRB1', 'CD3', 
                'DAPI9', 'p53', 'CD45RO', 'FOXA1', 'AMACR', 
                'DAPI10', 'CDX2', 'HOXB13', 'CD20', 'NOTCH1']
    '''
    channels = ['DAPI1','aSMA','Tryp','Ki67',
                'CD68','AR','CD20','ChromA','CK5',
                'HLADRB1','CD3','CD11b','CD4','CD45',
                'CD163','CD66b','PD1','GZMB','NKX31',
                'CK8','AMACR','FOXP3']
    
    keep_channels = [ch for ch in channels if (('DAPI' not in ch) or (('DAPI' in ch) and (ch == 'DAPI1'))) and (ch != 'c2')]
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx


def get_mask(sample_name):
    mask_dir = "/home/groups/CEDAR/panel-reduction-project/ACED-immune/mesmer_cell_mask_tissue_loss"
    fname = [f for f in os.listdir(mask_dir) if (sample_name in f) and ('cell_mask' in f)][0]
    return imread(f'{mask_dir}/{fname}')
    
    
def get_image(sid):
    #fname = f'/home/groups/CEDAR/panel-reduction-project/ACED-immune/ACED-IMMUNE-{sid}.ome.tiff'
    fname = f'/home/groups/ChangLab/dataset/ACED-IMMUNE-Normalized-ACED-Ref/ACED-IMMUNE-{sid}_normalized.ome.tiff'
    imdata = zarr.open(tifffile.imread(fname, aszarr=True))
    return da.from_zarr(imdata)



if __name__ == '__main__':
    print('entering main')
    CROP_SIZE = 64
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    
    save_dir =  '/home/groups/ChangLab/dataset/aced-immune-norm'
    if not os.path.exists(save_dir): os.mkdir(save_dir)  

    ids = ['22-2-A-2_scene003', '18-1-B-1of2-2_scene002', '15-1-A-2_scene008', '18-1-B-1of2-2_scene004',
           '17-3-B-2of2-2_scene002','01-1-A-1_scene005', '04-3-A-1_scene002', '01-2-A-1_scene001', '16-1-B-2_scene001',
           '17-3-B-2of2-2_scene000','28-1-B-2_scene004', '24-1-A-2of2-1_scene000', '17-2-A-2_scene001', '22-2-B-2_scene006',
           '22-2-B-2_scene003','15-1-A-2_scene014', '22-2-B-2_scene002', '17-2-A-2_scene000', '17-2-A-2_scene002',
           '01-1-A-1_scene000','18-1-B-1of2-2_scene003', '04-2-B-1_scene006', '22-2-A-2_scene004', '04-2-B-1_scene003',
           '01-1-A-1_scene007','26-1-B-1of2-2_scene002', '24-1-A-2of2-1_scene004', '15-1-A-2_scene012', '01-2-A-1_scene003',
           '24-1-A-2of2-1_scene002','09-1-A-1_scene003', '17-3-B-2of2-2_scene005', '15-1-A-2_scene005', '24-1-B-1_scene003',
           '24-1-A-2of2-1_scene003','26-1-A-2_scene004', '15-1-A-2_scene007', '26-1-B-1of2-2_scene001', '03-1-B-1of3-1_scene001',
           '26-1-B-1of2-2_scene003','01-1-A-1_scene002', '22-2-A-2_scene000', '09-1-A-1_scene002', '17-2-A-2_scene003',
           '17-3-B-2of2-2_scene001','15-1-A-2_scene006', '26-1-A-2_scene002', '22-2-B-2_scene000', '15-1-A-2_scene004',
           '04-2-B-1_scene005','01-1-A-1_scene008', '24-1-B-1_scene002', '15-1-A-2_scene011', '22-2-B-2_scene005',
           '09-1-A-1_scene004','16-1-B-2_scene000', '26-1-A-2_scene000', '24-1-A-2of2-1_scene005', '01-2-A-1_scene002',
           '28-1-B-2_scene003','24-1-B-1_scene001', '15-1-A-2_scene002', '22-2-B-2_scene001', '09-1-A-1_scene005',
           '22-2-A-2_scene001','23-1-B-2_scene000', '15-1-A-2_scene001', '22-2-A-2_scene002', '18-1-B-1of2-2_scene001',
           '26-1-A-2_scene003','28-1-B-2_scene002', '28-1-B-2_scene000', '15-1-A-2_scene010', '06-2-A-1_scene000',
           '22-2-A-2_scene005','15-1-A-2_scene003', '09-1-A-1_scene001', '26-1-A-2_scene001', '24-1-B-1_scene000',
           '28-1-B-2_scene005','04-2-B-1_scene001', '01-1-A-1_scene006', '01-1-A-1_scene004', '04-3-A-1_scene001',
           '15-1-A-2_scene000','24-1-A-2of2-1_scene001', '23-1-B-2_scene003', '04-2-B-1_scene002', '04-2-B-1_scene000',
           '26-1-B-1of2-2_scene000', '18-1-B-1of2-2_scene000', '22-2-B-2_scene007', '26-1-A-2_scene006', '01-2-A-1_scene004',
           '22-2-B-2_scene004', '01-1-A-1_scene009', '01-1-A-1_scene001', '16-1-B-2_scene002', '06-2-A-1_scene002',
           '06-2-A-1_scene001', '23-1-B-2_scene001', '28-1-B-2_scene001', '17-3-B-2of2-2_scene003', '23-1-B-2_scene002',
           '09-1-A-1_scene006', '04-2-B-1_scene004']
    
    for sample_name in ids:
        save_fname = f'aced_immune_dataset_norm_sid={sample_name}.h5'
            
        print('retrieving samples...')
        IF = get_image(sample_name)
        IF = IF[keep_channels_idx].compute()
        print(IF.shape)
        
        print('retrieving cell mask...')
        cell_mask = get_mask(sample_name)
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
            