import torch
import numpy as np
import pandas as pd
from model_panel_evaluation import get_model, get_dataloader
from intensity import get_intensities, get_spearman

def generate_imputed_feature_table(unmasked_ch_idx, sid, model_id, gpu_id, dataset_size, save_fname): 
    #model setup
    device = torch.device(f'cuda:{gpu_id}')
    ckpt_path = '/home/groups/ChangLab/sakiyama/cycif-panel-reduction/training/ORION-CRC'
    params_file = "params.json"
    model = get_model(model_id, ckpt_path, params_file, device)

    #data setup
    val_file = f'/mnt/scratch/ORION-CRC/orion_crc_dataset_sid={sid}.h5'
    BATCH_SIZE = 5_000
    remove_background = True
    dataloader = get_dataloader(val_file, BATCH_SIZE, dataset_size, remove_background)

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
            'E-cadherin',
            'PD-1',
            'Ki67',
            'PanCK',
            'aSMA']    
    ch2stain = {i:ch for i,ch in enumerate(channels)}
    NUM_CHANNELS = len(channels)

    calculate_ssims = False
    masked_ch_idx = [ch for ch in range(NUM_CHANNELS) if ch not in unmasked_ch_idx]
    mints, pmints, ssims, meta = get_intensities(model=model, 
                                           panel=unmasked_ch_idx, 
                                           val_loader=dataloader, 
                                           ch2stain=ch2stain,
                                           calculate_ssims=calculate_ssims,
                                           device=device,
                                           return_meta=True)
    def get_xy(meta_):
        x = str(meta_).split('-')[-2]
        y = str(meta_).split('-')[-1]

        x = int(x.split('=')[-1])
        y = int(y.split('=')[-1].replace("'", ""))
        return x,y

    coords = np.vectorize(get_xy)(meta)
    data = np.concatenate([pmints, np.expand_dims(coords[0],1), np.expand_dims(coords[1],1)], axis=1)


    df = pd.DataFrame(data, columns=[ch+'_imputed' for i,ch in enumerate(channels) if i in masked_ch_idx] +['X_centroid','Y_centroid'])
    df.to_csv(save_fname, index=False)
    
    
if __name__ == '__main__':
    unmasked_ch_idx = [0,1,2,3] #index of channels to remain unmasked
    sid = 'CRC05' #sample id to run on
    model_id = 'hxn3pdwd'
    gpu_id = 4
    dataset_size = 1_000_000
    save_fname = 'crc05_pmint_test.csv'
    
    generate_imputed_feature_table(unmasked_ch_idx, sid, model_id, gpu_id, dataset_size, save_fname)
    
    