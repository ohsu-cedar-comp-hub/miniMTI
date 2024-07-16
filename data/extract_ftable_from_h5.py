import h5py
import pandas as pd
import numpy as np
from einops import repeat 

path = '/home/groups/ChangLab/dataset/ORION-CRC/orion_crc_dataset_sid=CRC05.h5'
save_fname = 'crc05_test3.csv'
f = h5py.File(path)

ims = f['images'][:,:,:,:-2]
masks = f['masks'][:]
meta = f['metadata'][:]

masks = repeat(masks, 'n h w -> n h w c', c=17)
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

'''
channels = ['DAPI_R1','aSMA','Tryp','Ki67', 'CD68','AR','CD20','ChromA','CK5','HLADRB1','CD3',
                'CD11b','CD4','CD45','CD163','CD66b','PD1','GZMB','NKX31','CK8','AMACR','FOXP3','CD8',
                'EPCAM','CD56','NCR1','ERG','CK14','ECAD','VIM','FOSB','CD31','Tbr2','CD45RA','p53',
                'CD45RO','FOXA1','CDX2','HOXB13','NOTCH1']
'''
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
channels = [ch for ch in channels if (('DAPI' not in ch) or (('DAPI' in ch) and (ch == 'DAPI1'))) and (ch != 'c2')]
'''
df = pd.DataFrame(data, columns=channels+['X_centroid','Y_centroid'])
df.to_csv(save_fname, index=False)