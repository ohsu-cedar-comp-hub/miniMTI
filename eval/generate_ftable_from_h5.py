import torch
import numpy as np
import pandas as pd
from model_panel_evaluation import get_model, get_dataloader
from intensity import get_intensities, get_spearman

def generate_imputed_feature_table(model, dataloader, channels, unmasked_ch_idx, save_fname, device):     
    ch2stain = {i:ch for i,ch in enumerate(channels)}
    NUM_CHANNELS = len(channels)

    calculate_ssims = False
    masked_ch_idx = [i for i,ch in enumerate(channels) if ch not in unmasked_ch_idx]
    mints, pmints, ssims, meta = get_intensities(model=model, 
                                                 panel=unmasked_ch_idx, 
                                                 val_loader=dataloader, 
                                                 ch2stain=ch2stain,
                                                 calculate_ssims=calculate_ssims,
                                                 device=device,
                                                 return_meta=True
                                                )

    mints = mints[:, masked_ch_idx]
    pmints = pmints[:, masked_ch_idx]
    
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
    model_id = '12s3fyc6' #rgb mask H&E only
    ckpt_path = '/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction/'
    params_file = "params_mvtm-256-he-rgb.json"
    model = get_model(model_id, ckpt_path, params_file, device)
    
    #data setup
    sid = 'CRC05' #sample id to run on
    val_file = f'/mnt/scratch/ORION-CRC-Unnormalized-fixed-HE/orion_crc_dataset_sid={sid}.h5'
    batch_size = 256
    dataset_size = 1000
    downscale = False
    remove_he = False
    deconvolve_he = False
    rescale = True
    remove_background=False
    dataloader = get_dataloader(val_file, batch_size, dataset_size, downscale=downscale, remove_background=remove_background, remove_he=remove_he, rescale=rescale,deconvolve_he=deconvolve_he)

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
    if deconvolve_he:
        channels.extend(['Hematoxylin','Eosin'])
    else:
        channels.append('H&E')

    unmasked_ch_idx = [17] #index of channels to remain unmasked
    save_fname = f'{sid}_real_predicted_features_input=he_only_rgb_mask_he-test.csv'
    
    generate_imputed_feature_table(model, dataloader, channels, unmasked_ch_idx, save_fname, device)
    
    