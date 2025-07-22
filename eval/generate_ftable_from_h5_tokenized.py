import os
import sys
sys.path.append('../data')
from data import get_panel_selection_data
import torch
import json
import numpy as np
import pandas as pd
from intensity_tokenized import get_intensities, get_spearman
import gc

def get_ckpt(ckpt_id, ckpt_path):
    '''get model checkpoint'''
    ckpt_dir = f"{ckpt_path}/{ckpt_id}/checkpoints/"
    ckpt_paths = [f'{ckpt_dir}/{f}' for f in os.listdir(ckpt_dir)]
    ckpt_paths.sort(key=os.path.getmtime) #get most recent ckpt
    return ckpt_paths[-1]

def get_model_and_tokenizer(configs, device):
    '''load model weights and tokenizer and return both'''
    # Get model checkpoint
    model_ckpt = get_ckpt(configs['model_id'], configs['ckpt_path'])
    
    # Load tokenizer
    sys.path.append('../training/MVTM')
    from tokenizer import Tokenizer
    from mvtm import IF_MVTM
    
    # Load tokenizer (VQGAN)
    tokenizer = Tokenizer(**configs['tokenizer'])
    
    tokenizer.if_tokenizer = tokenizer.if_tokenizer.to(device)
    if configs['tokenizer']['he_config_path']:
        tokenizer.he_tokenizer = tokenizer.he_tokenizer.to(device)       
    
    # Load model (MVTM)
    from mvtm_tokenized import Tokenized_MVTM
    model = Tokenized_MVTM(**configs['model']).load_from_checkpoint(model_ckpt, **configs['model'])
    
    return model.to(device).eval(), tokenizer

def get_dataloader(val_file, batch_size, dataset_size, downscale=False, remove_background=False, remove_he=False, shuffle=False, rescale=True, deconvolve_he=True):
    '''get data, returns pytorch dataloader'''
    if type(val_file) == str: val_file = [val_file]
    return get_panel_selection_data(val_file=val_file, batch_size=batch_size, dataset_size=dataset_size, remove_background=remove_background, remove_he=remove_he, downscale=downscale, shuffle=shuffle, rescale=rescale, deconvolve_he=deconvolve_he) 

def generate_imputed_feature_table(model, tokenizer, dataloader, channels, unmasked_ch_idx, save_fname, device, T=1):     
    ch2stain = {i:ch for i,ch in enumerate(channels)}
    NUM_CHANNELS = len(channels)

    masked_ch_idx = [i for i,ch in enumerate(channels) if i not in unmasked_ch_idx]
    mints, pmints, ssims, meta = get_intensities(
        model=model, 
        tokenizer=tokenizer,
        panel=unmasked_ch_idx, 
        val_loader=dataloader, 
        ch2stain=ch2stain,
        calculate_ssims=False,
        device=device,
        return_meta=True,
        T=T
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
    print(f"Saved results to {save_fname}")
    
    # Clear memory
    del mints, pmints, data, df
    gc.collect()
    torch.cuda.empty_cache()
    
    
if __name__ == '__main__':
    # Device setup
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    
    # Model setup
    config_path = "configs/params_mvtm-256-he-rgb-tokenized.json"  # params for tokenized model
    with open(config_path,"r") as f:
        configs = json.load(f)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(configs, device)
    
    # Data setup
    id_ = sys.argv[1]
    sid = f'CRC{id_}'  # sample id to run on
    val_file = f'/home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All/orion_crc_dataset_sid={sid}.h5'
    batch_size = 64
    dataset_size = 100_000
    
    # Load data
    dataloader = get_dataloader(val_file, batch_size, dataset_size, **configs['data'])
    
    # Set channels
    from crc_orion_channel_info import get_channel_info
    channels,_,_ = get_channel_info()  
    
    if configs['data']['deconvolve_he']:
        channels.extend(['Hematoxylin','Eosin'])
    else:
        channels.append('H&E')
        
    # Set unmasked channel indices
    unmasked_ch_idx = [17, 6, 11, 13]  # index of channels to remain unmasked
    
    # Set number of decoding steps
    T = 1
    
    # Define output filename
    save_fname = f'{sid}_real_predicted_features_input={len(unmasked_ch_idx)}_mx_rev_rgb_model_id={configs["model_id"]}_all_{T=}.csv'
    
    # Generate imputed feature table
    generate_imputed_feature_table(
        model, 
        tokenizer, 
        dataloader, 
        channels, 
        unmasked_ch_idx, 
        save_fname, 
        device, 
        T
    )