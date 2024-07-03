import h5py
import pandas as pd
import numpy as np
from einops import repeat 

path = '/mnt/scratch/ORION-CRC/orion_crc_dataset_sid=CRC05.h5'
save_fname = 'CR05_processed_feature_table.csv'
f = h5py.File(path)

ims = f['images'][:]
masks = f['masks'][:]
meta = f['metadata'][:]

masks = repeat(masks, 'n h w -> n h w c', c=19)
mints =(ims * masks).sum(axis=(1,2)) / masks.sum(axis=(1,2))

def get_xy(meta_):
    x = str(meta_).split('-')[-2]
    y = str(meta_).split('-')[-1]
    
    x = int(x.split('=')[-1])
    y = int(y.split('=')[-1].replace("'", ""))
    return x,y

coords = np.vectorize(get_xy)(meta)

data = np.concatenate([mints, np.expand_dims(coords[0],1), np.expand_dims(coords[1],1)], axis=1)

channels = [
'DAPI',
'AF1',
'CD31',
'CD45',
'CD68',
'Argo550',
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

df = pd.DataFrame(data, columns=channels+['X_centroid','Y_centroid'])
df.to_csv(save_fname)