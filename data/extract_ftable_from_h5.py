import h5py
import os
import pandas as pd
import numpy as np
from einops import repeat 
from tqdm import tqdm


dir_ = '/home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All'
save_dir = '/home/groups/cedar-acedreimagine/ORION-CRC-gt-ftables'

for fname in tqdm(os.listdir(dir_)):
    path = f"{dir_}/{fname}"
    if "=CRC" not in path: continue
    f = h5py.File(path)

    ims = f['images'][:,:,:,:-3]
    masks = f['masks'][:]
    meta = f['metadata'][:]

    masks = repeat(masks, 'n h w -> n h w c', c=43)
    mints =(ims * masks).sum(axis=(1,2)) / masks.sum(axis=(1,2))

    def get_xy(meta_):
        x = str(meta_).split('-')[-2]
        y = str(meta_).split('-')[-1]

        x = int(x.split('=')[-1])
        y = int(y.split('=')[-1].replace("'", ""))
        return x,y

    coords = np.vectorize(get_xy)(meta)

    data = np.concatenate([mints, np.expand_dims(coords[0],1), np.expand_dims(coords[1],1)], axis=1)

    '''
    channels = [
    'DAPI',
    #'AF1',
    'CD31',
    'CD45',
    'CD68',
    #'Argo550',
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

    '''
    channels = ['DAPI','TRITC','Cy5','TOMM20','CD90','CD45','ERG','HLA-DR','CD11b','CD3','AR','TUBB3','GZMB',
                'Ecad','CK5','CD68','TH','aSMA','AMACR','CD56','NFKB','HIF-1','CD4','FOXA1','ADAM10','DCX',
                'CD11c','CD20','Ki67','CD8','CD31','VIM','NRXN1','NLGN4X','ChromA','TRYP','CD44','NLGN1','CK8',
                'B-catenin','H3K4','H3K27ac','CD163']
    '''
    channels = ['DAPI','aSMA','Tryp','Ki67', 'CD68','AR','CD20','ChromA','CK5','HLADRB1','CD3',
                    'CD11b','CD4','CD45','CD163','CD66b','PD1','GZMB','NKX31','CK8','AMACR','FOXP3','CD8',
                    'EPCAM','CD56','NCR1','ERG','CK14','ECAD','VIM','FOSB','CD31','Tbr2','CD45RA','p53',
                    'CD45RO','FOXA1','CDX2','HOXB13','NOTCH1']

    
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
    channels = [ch for ch in channels if (('DAPI' not in ch) or (('DAPI' in ch) and (ch == 'DAPI1'))) and (ch != 'c2')]
    '''
    df = pd.DataFrame(data, columns=channels+['X_centroid','Y_centroid'])
    save_fname = f"{save_dir}/{fname.replace('.h5','.csv')}"
    df.to_csv(save_fname, index=False)
