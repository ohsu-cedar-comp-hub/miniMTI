import os
import sys
import ast
import json
import torch
import numpy as np
#from eval_mae import IF_MAE
from intensity import get_intensities, get_spearman
from plotting import plot_scatter, plot_heatmap, plot_violin, plot_hist
sys.path.append('../data')
from data import get_panel_selection_data
sys.path.append('../training/MVTM')
from mvtm import IF_MVTM

def get_ckpt(ckpt_id, ckpt_path):
    '''get model checkpoint'''
    dir_ = f"{ckpt_path}/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    return f"{dir_}/{fname}"

def get_model(model_id, ckpt_path, params_file, device, model_type='mvtm'):
    '''load model weights and return model'''
    with open(params_file) as f:
        params = json.load(f)
    ckpt = get_ckpt(model_id, ckpt_path)
    if model_type == 'mvtm':
        model = IF_MVTM(**params).load_from_checkpoint(ckpt, **params)
    else:
        model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    return model.to(device).eval()

def get_marker_order(order_path):
    '''get ordered list of markers generated from panel selection'''
    with open(order_path) as f:
        order = f.readlines()[0]
    return ast.literal_eval(order)
    
def get_dataloader(val_file, batch_size, dataset_size, downscale=False, remove_background=False, remove_he=False, shuffle=False):
    '''get data, returns pytorch dataloader'''
    if type(val_file) == str: val_file = [val_file]
    return get_panel_selection_data(val_file=val_file, batch_size=batch_size, dataset_size=dataset_size, remove_background=remove_background, remove_he=remove_he, downscale=downscale, shuffle=shuffle) 
    
def evaluate(model, dataloader, marker_order, save_id, device):
    '''run model inference on all panel sizes and generate figures''' 
    #load channel info
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    ch2stain = {i:ch for ch,i in ch2idx.items()}

    #for each panel size: get real and predicted mean intensities, calculate correlation,  plot scatterplot for each marker
    panel_sizes = [i for i in range(1, len(marker_order) + 1)]
    panel_sizes = [14]
    corrs_per_panel_size = []
    for i,size in enumerate(panel_sizes):
        #get masked and unmasked channel indices
        #unmasked_ch_idx = marker_order[:size]
        unmasked_ch_idx = [0,1,2,3,14,15,16] #orion
        #unmasked_ch_idx = [0,1,10,13,28,31] #aced
        masked_ch_idx = [i for i in range(NUM_CHANNELS) if i not in unmasked_ch_idx]
        #get real and predicted mean intensities and SSIM scores for each sample for the panel size
        mints, pmints, ssims = get_intensities(model=model, 
                                               panel=unmasked_ch_idx, 
                                               val_loader=dataloader, 
                                               ch2stain=ch2stain,
                                               calculate_ssims=calculate_ssims,
                                               device=device)
        #calculate spearman correlations per marker between real and predicted mean intensities
        corrs_per_marker = get_spearman(mints, pmints)
        corrs_per_panel_size.append(corrs_per_marker)
        #plot scatterplot and histogram of real and predicted mean intensities
        plot_scatter(mints, pmints, corrs_per_marker, masked_ch_idx, ch2stain, save_id)
        plot_hist(mints, pmints, masked_ch_idx, ch2stain, save_id)
        np.save(f'correlations/corrs_per_panel_size_{size}_{save_id}.npy', corrs_per_marker.detach().cpu().numpy())

    #save spearman correlations per marker per panel size
    corrs_per_panel_size = [s.detach().cpu().numpy() for s in corrs_per_panel_size]
    np.save(f'correlations/corrs_per_panel_size_{save_id}.npy', corrs_per_panel_size)

    plot_heatmap(corrs_per_panel_size, marker_order, NUM_CHANNELS, ch2stain, save_id)
    plot_violin(corrs_per_panel_size, NUM_CHANNELS, save_id)


if __name__ == '__main__':
    calculate_ssims = False
    device = torch.device('cuda:1')
    BATCH_SIZE = 100
    DATASET_SIZE = 1_000
    dataset = 'orion'
    
    if dataset == 'aced':
        NUM_CHANNELS = 40
        from aced_stitch2_channel_info import get_channel_info
        model_id = 'e37xf8r1'
        val_id = '23-1-B-2_scene000'
        downscale = True
        remove_he = False
        params_file="params_mvtm_aced.json"
        val_file = '/mnt/scratch/aced-immune-norm-40mx/aced_immune_dataset_norm_sid=23-1-B-2_scene000.h5'
        
    elif dataset == 'orion':
        NUM_CHANNELS = 17
        from crc_orion_channel_info import get_channel_info
        model_id = 'q6cn0ooj'
        val_sid = 'CRC05-all'
        downscale = False
        remove_he = True
        params_file="params_mvtm.json"
        val_file = '/mnt/scratch/ORION-CRC-Unnormalized/orion_crc_dataset_sid=CRC05.h5'
        
    ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction/'
    panel_selection_process = 'human'
    save_id = f'{model_id}-{panel_selection_process}-panel-val-sid={val_sid}-50k-mvtm'
    order_path = 'orderings/train-batch6-out.h5_xwabaemb_panel_order_ips.txt'
    if not os.path.exists(f'plots/{save_id}'): os.mkdir(f'plots/{save_id}')
        
    model = get_model(model_id, ckpt_path, params_file, device)
    dataloader = get_dataloader(val_file, BATCH_SIZE, DATASET_SIZE, downscale=downscale, remove_background=False, remove_he=remove_he)
    marker_order = get_marker_order(order_path)
    
    evaluate(model, dataloader, marker_order, save_id, device)
    
