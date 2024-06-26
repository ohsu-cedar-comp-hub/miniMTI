import os
import sys
import ast
import json
import torch
import numpy as np
from eval_mae import IF_MAE
from intensity import get_intensities, get_spearman
from plotting import plot_scatter, plot_heatmap, plot_violin, plot_hist
sys.path.append('../data')
from data import get_panel_selection_data
#from process_cedar_biolib_immune import get_channel_info
from process_aced_immune import get_channel_info

def get_ckpt(ckpt_id, ckpt_path):
    '''get model checkpoint'''
    dir_ = f"{ckpt_path}/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    return f"{dir_}/{fname}"

def get_model(model_id, ckpt_path, params_file, device):
    '''load model weights and return model'''
    with open(params_file) as f:
        params = json.load(f)
    ckpt = get_ckpt(model_id, ckpt_path)
    model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    return model.to(device).eval()

def get_marker_order(order_path):
    '''get ordered list of markers generated from panel selection'''
    with open(order_path) as f:
        order = f.readlines()[0]
    return ast.literal_eval(order)
    
def get_dataloader(val_file, batch_size, dataset_size, remove_background):
    '''get data, returns pytorch dataloader'''
    return get_panel_selection_data(val_file=[val_file], batch_size=batch_size, dataset_size=dataset_size, remove_background=remove_background) 
    
def evaluate(model, dataloader, marker_order, save_id, device):
    '''run model inference on all panel sizes and generate figures''' 
    #load channel info
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    ch2stain = {i:ch for ch,i in ch2idx.items()}

    #for each panel size: get real and predicted mean intensities, calculate correlation,  plot scatterplot for each marker
    panel_sizes = [i for i in range(1, len(marker_order) + 1)]
    #panel_sizes = [11]
    corrs_per_panel_size = []
    for i,size in enumerate(panel_sizes):
        #get masked and unmasked channel indices
        unmasked_ch_idx = marker_order[:size]
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
        #plot_scatter(mints, pmints, corrs_per_marker, masked_ch_idx, ch2stain, save_id)
        #plot_hist(mints, pmints, masked_ch_idx, ch2stain, save_id)

    #save spearman correlations per marker per panel size
    corrs_per_panel_size = [s.detach().cpu().numpy() for s in corrs_per_panel_size]
    np.save(f'correlations/corrs_per_panel_size_{save_id}.npy', corrs_per_panel_size)

    plot_heatmap(corrs_per_panel_size, marker_order, NUM_CHANNELS, ch2stain, save_id)
    plot_violin(corrs_per_panel_size, NUM_CHANNELS, save_id)


if __name__ == '__main__':
    remove_background = True
    calculate_ssims = False
    device = torch.device('cuda:1')
    BATCH_SIZE = 1000
    NUM_CHANNELS = 22
    DATASET_SIZE = 100_000
    #model_id = 'd3ps83ub' #40 marker
    model_id = '7u2ixfa8' #22 marker norm
    #model_id = 'djqmufd1' #22 marker unnorm
    ckpt_path = '../training/cedar-panel-reduction'
    params_file="params.json"
    #val_sid = '15-1-A-2_scene004'
    val_sid = '09-1-A-1_scene004'
    #val_sid = '18-1-B-1of2-2_scene002'
    #val_sid = '18-1-B-1of2-2_scene002'
    panel_selection_process = 'ips-reverse'
    save_id = f'{model_id}-{panel_selection_process}-panel-val-sid={val_sid}'
    val_file = f'/mnt/scratch/aced-immune-norm/aced_immune_dataset_norm_sid={val_sid}.h5'
    order_path = 'orderings/train-batch-2-out.h5_7u2ixfa8_panel_order_ips-reversed.txt'
    if not os.path.exists(f'plots/{save_id}'): os.mkdir(f'plots/{save_id}')
        
    model = get_model(model_id, ckpt_path, params_file, device)
    dataloader = get_dataloader(val_file, BATCH_SIZE, DATASET_SIZE, remove_background)
    marker_order = get_marker_order(order_path)
    
    evaluate(model, dataloader, marker_order, save_id, device)
    
