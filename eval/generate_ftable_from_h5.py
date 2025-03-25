import os
import sys
sys.path.append('../data')
from data import get_panel_selection_data
import torch
import json
import numpy as np
import pandas as pd
from intensity import get_intensities, get_spearman

def get_ckpt(ckpt_id, ckpt_path):
    '''get model checkpoint'''
    dir_ = f"{ckpt_path}/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    return f"{dir_}/{fname}"

def get_model(model_id, ckpt_path, params_file, device, model_type='mvtm'):
    '''load model weights and return model'''
    with open(params_file) as f:
        params = json.load(f)
    params['cls_token'] = False
    ckpt = get_ckpt(model_id, ckpt_path)
    if model_type == 'mvtm':
        sys.path.append('../training/MVTM')
        from mvtm import IF_MVTM
        model = IF_MVTM(**params).load_from_checkpoint(ckpt, **params)
    else:
        from eval_mae import IF_MAE
        model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    return model.to(device).eval()
    
def get_dataloader(val_file, batch_size, dataset_size, downscale=False, remove_background=False, remove_he=False, shuffle=False, rescale=True, deconvolve_he=True):
    '''get data, returns pytorch dataloader'''
    if type(val_file) == str: val_file = [val_file]
    return get_panel_selection_data(val_file=val_file, batch_size=batch_size, dataset_size=dataset_size, remove_background=remove_background, remove_he=remove_he, downscale=downscale, shuffle=shuffle, rescale=rescale, deconvolve_he=deconvolve_he) 

def generate_imputed_feature_table(model, dataloader, channels, unmasked_ch_idx, save_fname, device):     
    ch2stain = {i:ch for i,ch in enumerate(channels)}
    NUM_CHANNELS = len(channels)

    calculate_ssims = False
    masked_ch_idx = [i for i,ch in enumerate(channels) if i not in unmasked_ch_idx]
    mints, pmints, ssims, meta = get_intensities(model=model, 
                                                 panel=unmasked_ch_idx, 
                                                 val_loader=dataloader, 
                                                 ch2stain=ch2stain,
                                                 calculate_ssims=calculate_ssims,
                                                 device=device,
                                                 return_meta=True
                                                )
    
    def get_xy(meta_):
        x = str(meta_).split('-')[-2]
        y = str(meta_).split('-')[-1]

        x = int(x.split('=')[-1])
        y = int(y.split('=')[-1].replace("'", ""))
        return x,y

    coords = np.vectorize(get_xy)(meta)
    data = np.concatenate([mints, pmints, np.expand_dims(coords[0],1), np.expand_dims(coords[1],1)], axis=1)

    df = pd.DataFrame(data, columns=[ch for i,ch in enumerate(channels) if i in masked_ch_idx] + [ch+'_imputed' for i,ch in enumerate(channels) if i in masked_ch_idx] +['X_centroid','Y_centroid'])
    
    df.to_csv(save_fname, index=False)
    
    
if __name__ == '__main__':
    #device setup
    gpu_id = 6
    device = torch.device(f'cuda:{gpu_id}')
    #model setup
    #model_id = 'hf6d7cfy' #rgb mask H&E only
    #model_id = '9x3324j4'
    model_id = 'ota9pmon'
    ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction/'
    #params_file = "params_mvtm-256-he-rgb.json" #orion
    params_file = 'params_mvtm-256-lunaphore.json'
    model = get_model(model_id, ckpt_path, params_file, device)
    
    #data setup
    #sid = 'CRC06' #sample id to run on
    sid = '1010178'
    #val_file = f'/mnt/scratch/ORION-CRC-Unnormalized-fixed-HE/orion_crc_dataset_sid={sid}.h5'
    val_file = f'/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid={sid}.h5'
    batch_size = 32
    dataset_size = 1_000_000
    downscale = True #true for prostate models
    remove_he = False
    deconvolve_he = False
    rescale = True
    remove_background=False
    dataloader = get_dataloader(val_file, batch_size, dataset_size, downscale=downscale, remove_background=remove_background, remove_he=remove_he, rescale=rescale,deconvolve_he=deconvolve_he)
    '''
    #set channels
    channels = ['DAPI',
            'CD31',
            'CD45',
            'CD68',
            'CD4',
            'FOXP3',
            'CD8a',
            'CD45RO',
            'CD20',
            'PD-L1',
            'CD3e', #Cd3?
            'CD163',
            'Ecad',
            'PD-1',
            'Ki67',
            'PanCK',
            'aSMA']  
    '''
    channels = ['DAPI','TRITC','Cy5','TOMM20','CD90','CD45','ERG','HLA-DR','CD11b','CD3','AR','TUBB3','GZMB',
                'Ecad','CK5','CD68','TH','aSMA','AMACR','CD56','NFKB','HIF-1','CD4','FOXA1','ADAM10','DCX',
                'CD11c','CD20','Ki67','CD8','CD31','VIM','NRXN1','NLGN4X','ChromA','TRYP','CD44','NLGN1','CK8',
                'B-catenin','H3K4','H3K27ac','CD163']
    '''
    if deconvolve_he:
        channels.extend(['Hematoxylin','Eosin'])
    else:
        channels.append('H&E')
    '''
    #unmasked_ch_idx = [17] #index of channels to remain unmasked
    unmasked_ch_idx = [0,33,16,30]
    #save_fname = f'{sid}_real_predicted_features_input=he_only_rgb_mask_he-only_model_id={model_id}.csv'
    save_fname = f'lunaphore_{sid}_real_predicted_features_input=4_mx_model_id={model_id}.csv'
    
    generate_imputed_feature_table(model, dataloader, channels, unmasked_ch_idx, save_fname, device)
    
    