import os
import sys
import ast
import json
import torch
import numpy as np
from intensity import get_intensities, get_spearman
from plotting import plot_scatter, plot_heatmap, plot_violin, plot_hist
sys.path.append('../data')
from data import get_panel_selection_data


def get_ckpt(ckpt_id, ckpt_path):
    '''get model checkpoint'''
    dir_ = f"{ckpt_path}/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    return f"{dir_}/{fname}"

def get_model(model_id, ckpt_path, params_file, device, model_type='mvtm'):
    '''load model weights and return model'''
    with open(params_file) as f:
        params = json.load(f)
    #params['cls_token'] = True
    ckpt = get_ckpt(model_id, ckpt_path)
    if model_type == 'mvtm':
        sys.path.append('../training/MVTM')
        from mvtm import IF_MVTM
        model = IF_MVTM(**params).load_from_checkpoint(ckpt, **params)
    else:
        from eval_mae import IF_MAE
        model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    return model.to(device).eval()

def get_marker_order(order_path):
    '''get ordered list of markers generated from panel selection'''
    with open(order_path) as f:
        order = f.readlines()[0]
    return ast.literal_eval(order)
    
def get_dataloader(val_file, batch_size, dataset_size, downscale=False, remove_background=False, remove_he=False, shuffle=False, rescale=True, deconvolve_he=True):
    '''get data, returns pytorch dataloader'''
    if type(val_file) == str: val_file = [val_file]
    return get_panel_selection_data(val_file=val_file, batch_size=batch_size, dataset_size=dataset_size, remove_background=remove_background, remove_he=remove_he, downscale=downscale, shuffle=shuffle, rescale=rescale, deconvolve_he=deconvolve_he) 
    
def evaluate(model, dataloader, marker_order, save_id, device, model_type='MVTM'):
    '''run model inference on all panel sizes and generate figures''' 
    #load channel info
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    ch2stain = {i:ch for ch,i in ch2idx.items()}

    #for each panel size: get real and predicted mean intensities, calculate correlation,  plot scatterplot for each marker
    #panel_sizes = [i for i in range(1, len(marker_order) + 1)]      
    #panel_sizes = [5,8,11,14]
    panel_sizes = [4,7,10,13]
    #panel_sizes=[15]
    corrs_per_panel_size = []
    calculate_ssims = True
    for i,size in enumerate(panel_sizes):
        #get masked and unmasked channel indices
        unmasked_ch_idx = marker_order[:size]
        masked_ch_idx = [i for i in range(len(keep_channels)) if i not in unmasked_ch_idx]
        #masked_ch_idx = [10, 16, 17, 22, 23, 25, 26, 27, 30, 33, 36, 37]
        #get real and predicted mean intensities and SSIM scores for each sample for the panel size
        mints, pmints, ssims = get_intensities(model=model, 
                                               panel=unmasked_ch_idx, 
                                               val_loader=dataloader, 
                                               ch2stain=ch2stain,
                                               calculate_ssims=calculate_ssims,
                                               device=device,
                                               model_type=model_type)
        #calculate spearman correlations per marker between real and predicted mean intensities
        corrs_per_marker = get_spearman(mints, pmints)
        corrs_per_panel_size.append(corrs_per_marker)
        print(f"mean correlation: {corrs_per_marker.detach().cpu().numpy().mean()}")
        np.save(f'correlations/corrs_per_panel_size_{size}_{save_id}.npy', corrs_per_marker.detach().cpu().numpy())
        np.save(f'ssims/ssims_{size}_{save_id}.npy', ssims.detach().numpy())
        #plot scatterplot and histogram of real and predicted mean intensities
        #plot_scatter(mints, pmints, corrs_per_marker, masked_ch_idx, ch2stain, save_id)
        #plot_hist(mints, pmints, masked_ch_idx, ch2stain, save_id)

    #save spearman correlations per marker per panel size
    #corrs_per_panel_size = [s.detach().cpu().numpy() for s in corrs_per_panel_size]
    #np.save(f'correlations/corrs_per_panel_size_{save_id}.npy', corrs_per_panel_size)

    #plot_heatmap(corrs_per_panel_size, marker_order, NUM_CHANNELS, ch2stain, save_id)
    #plot_violin(corrs_per_panel_size, NUM_CHANNELS, save_id)


if __name__ == '__main__':
    device = torch.device('cuda:7')
    BATCH_SIZE = 100
    DATASET_SIZE = 10000
    dataset = 'orion'
    model_type = 'mvtm'
    remove_background=False
    deconvolve_he=True

    if model_type == 'MAE':
        if dataset == 'orion':
            NUM_CHANNELS = 17
            from crc_orion_channel_info import get_channel_info
            #model_id = '8x90uhwo' #og
            model_id = 'y56gjgto' #og w/ background
            #model_id = '6vtoavjz'
            #model_id = 'jj7n5ftx' #w/o background
            #model_id = '0unn0es3' #og w/ background large
            remove_background=False
            val_sid = 'CRC14-all'
            downscale = False
            remove_he = True
            rescale=False
            params_file="params-mae.json"
            val_file = '/mnt/scratch/ORION-CRC-Unnormalized/orion_crc_dataset_sid=CRC01.h5'
            ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/cedar-panel-reduction/'
            
        if dataset == 'aced':
            NUM_CHANNELS = 40
            from aced_stitch2_channel_info import get_channel_info
            model_id = 'ajkrrllx'
            val_sid = 'batch6-all'
            remove_background=False
            downscale = True
            remove_he = False
            rescale = False
            deconvolve_he=False
            params_file="params_mae-aced.json"
            #val_file = '/mnt/scratch/aced-immune-norm-40mx/aced_immune_dataset_norm_sid=24-1-B-1_scene003.h5'
            #val_file = '/mnt/scratch/aced-immune-norm-40mx/val-batch6.h5'
            ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/cedar-panel-reduction/'
            
            
    else: #MVTM
        if dataset == 'aced':
            NUM_CHANNELS = 43
            #from aced_stitch2_channel_info import get_channel_info
            from lunaphore_channel_info import get_channel_info
            #model_id = 'jhxg3cma'
            #model_id = 'r485d56l' #trained on aced
            model_id = 'ota9pmon' #trained on lunaphore
            #val_sid = 'batch6-all'
            #val_sid = 'Lun-1010158'
            val_sid = '24-1-B-1_scene000'
            downscale = True
            remove_he = False
            rescale = True
            deconvolve_he=False
            #params_file="params_mvtm-256-aced.json"
            params_file="params_mvtm-256-lunaphore.json"
            #val_file = '/mnt/scratch/aced-immune-norm-40mx/aced_immune_dataset_norm_sid=24-1-B-1_scene003.h5'
            #val_file = '/mnt/scratch/aced-immune-norm-40mx/val-batch6.h5'
            #val_file = '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid=1010158.h5'
            val_file = '/mnt/scratch/aced-immune-norm-40mx/aced_immune_dataset_norm_sid=24-1-B-1_scene000.h5'
            ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction/'
            #ckpt_path = '/home/groups/CEDAR/kaoutar/projects/cycif-panel-reduction-master/MVTM-panel-reduction/'
            
        elif dataset == 'orion':
            NUM_CHANNELS = 18
            from crc_orion_channel_info import get_channel_info
            #model_id = 'kx7ev423' #1024
            #model_id = 's6y4l6t8' #256
            #model_id = 'kkeg5l99' #256 full-ch-mask
            #model_id = 'affyb41u' #1024 HE
            #model_id = 'hhosisn1'
            #model_id = 'a43cdi13'
            #model_id = '1pw3fbst'
            #model_id = 'rz21uadh'
            #model_id = 'amxyil5z'
            #model_id = 'ykqiuzjn'
            #model_id = 'a8bu5rxj'
            #model_id = 'htv9xax0'
            #model_id = '4b58c44t'
            #model_id = 'jn5m1o6j'
            #model_id = 'lmzdxiv4'
            #model_id = 'w51sp58j'
            #model_id = 'u4kmtzl6'
            #model_id = '2zdkptgq' #rgb
            #model_id = 'hoy7afh8' #deconvolved
            #model_id = '9x3324j4'
            #model_id = '64nq5vme'
            #model_id = 'm5cozryd'
            #model_id = 'e78ev5xs'
            model_id = '7qk7l4ku'
            val_sid = 'CRC05'
            downscale = False
            remove_he = False
            deconvolve_he = False
            rescale = True
            params_file="params_mvtm-256-he-rgb.json"
            #val_file = '/mnt/scratch/ORION-CRC-Unnormalized/orion_crc_dataset_sid=CRC05.h5'
            val_file = '/mnt/scratch/ORION-CRC-Unnormalized-fixed-HE/orion_crc_dataset_sid=CRC05.h5'
            #val_file = '/mnt/scratch/ChangLab/orion_crc_dataset_sid=CRC05.h5'
            #val_file = '/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized-Clip-Int8/orion_crc_dataset_sid=CRC05.h5'
            ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction/'
        
    
    panel_selection_process = 'random'
    save_id = f'{model_id}-{panel_selection_process}-panel-val-sid={val_sid}-{DATASET_SIZE=}-mvtm'
    #order_path = 'orderings/train-CRC05-06-out.h5_kkeg5l99_panel_order_ips.txt'
    #order_path = 'orderings/train-CRC05-06-out.h5_y56gjgto_panel_order_ips-reversed.txt'
    #order_path = 'orderings/train-CRC05-06-out.h5_amxyil5z_panel_order_ips-reversed.txt'
    #order_path = 'orderings/train-CRC05-06-out.h5_rz21uadh_panel_order_ips.txt'
    #order_path = 'orderings/train-batch6-out.h5_r485d56l_panel_order_ips.txt'
    #order_path = 'orderings/train-batch6-out.h5_ajkrrllx_panel_order_ips.txt'
    #order_path = 'orderings/orion_panel_select_data.h5_9x3324j4_panel_order_ips.txt'
    #order_path = 'orderings/orion_panel_select_data.h5_2zdkptgq_panel_order_ips-reversed.txt'
    order_path = 'orderings/train-CRC05-06-out.h5_2zdkptgq_panel_order_ips.txt'
    if not os.path.exists(f'plots/{save_id}'): os.mkdir(f'plots/{save_id}')
        
    model = get_model(model_id, ckpt_path, params_file, device, model_type=model_type)
    dataloader = get_dataloader(val_file, BATCH_SIZE, DATASET_SIZE, downscale=downscale, remove_background=remove_background, remove_he=remove_he, rescale=rescale,deconvolve_he=deconvolve_he)
    marker_order = get_marker_order(order_path)
    '''
    evaluate(model, dataloader, marker_order, save_id, device, model_type=model_type)
    '''
    #lunaphore_order = [0, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 22, 23, 27, 28, 29, 30, 31, 34, 35, 38, 42]
    #aced_order = [0, 13, 26, 9, 11, 10, 5, 17, 28, 8, 4, 1, 20, 24, 12, 36, 6, 3, 22, 31, 29, 7, 2, 19, 14]
    #evaluate(model, dataloader, lunaphore_order, save_id, device, model_type=model_type)
    '''
    '''
    random_orders = [
        [7, 5, 1, 14, 13, 11, 12, 15, 9, 2, 6, 8, 3, 4, 10, 16],
        [14, 9, 4, 6, 12, 11, 1, 2, 3, 10, 13, 8, 5, 16, 7, 15],
        [15, 11, 1, 12, 8, 2, 16, 14, 4, 7, 10, 3, 5, 6, 9, 13],
        [13, 4, 7, 14, 3, 10, 15, 8, 12, 9, 6, 16, 1, 5, 2, 11],
        [4, 3, 5, 6, 7, 11, 2, 13, 1, 10, 15, 16, 9, 14, 12, 8],
        [11, 12, 3, 7, 9, 5, 15, 16, 10, 8, 4, 1, 13, 14, 6, 2],
        [3, 10, 7, 5, 2, 1, 16, 13, 15, 11, 8, 9, 12, 4, 14, 6],
        [2, 1, 8, 11, 3, 15, 13, 16, 4, 7, 9, 5, 12, 14, 10, 6],
        [6, 16, 5, 1, 10, 13, 12, 14, 9, 15, 8, 3, 11, 2, 4, 7],
        [2, 9, 6, 14, 13, 10, 16, 11, 4, 1, 12, 8, 5, 3, 15, 7]
    ]
    for i,order in enumerate(random_orders):
        order = [0,17] + order
        if not os.path.exists(f'plots/{save_id}run-{i+1}'): os.mkdir(f'plots/{save_id}run-{i+1}')
        evaluate(model, dataloader, order, save_id+f"run-{i+1}", device, model_type=model_type)
        #evaluate(model, dataloader, order, save_id, device, model_type=model_type)
    
    
