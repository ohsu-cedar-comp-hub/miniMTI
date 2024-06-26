import os
import h5py
import numpy as np

train_files = []
val_files = [] 
data_dir = '/mnt/scratch/aced-immune-norm'
val_samples = ['09-1-A-1_scene001','09-1-A-1_scene002','09-1-A-1_scene003','09-1-A-1_scene004','09-1-A-1_scene005',
               '09-1-A-1_scene006','15-1-A-2_scene000','15-1-A-2_scene001','15-1-A-2_scene002','15-1-A-2_scene003',
               '15-1-A-2_scene004','15-1-A-2_scene005','15-1-A-2_scene006','15-1-A-2_scene007','15-1-A-2_scene008',
               '15-1-A-2_scene010','15-1-A-2_scene011','15-1-A-2_scene012','15-1-A-2_scene014','18-1-B-1of2-2_scene000',
               '18-1-B-1of2-2_scene001','18-1-B-1of2-2_scene002','18-1-B-1of2-2_scene003','18-1-B-1of2-2_scene004']
#batch_name = val_samples[0]
batch_name = 'batch-2'
    
for h5file in os.listdir(data_dir):
    if all([s not in h5file for s in val_samples]) and 'train' not in h5file:
        train_files.append(h5py.File(f"{data_dir}/{h5file}", "r"))
    else:
        val_files.append(h5py.File(f"{data_dir}/{h5file}", "r"))

num_train_samples = sum([f['images'].shape[0] for f in train_files])
im_shape = train_files[0]['images'].shape[1:]

with h5py.File(f'{data_dir}/train-{batch_name}-out.h5',mode='w') as h5w:
    h5w.create_dataset('images', shape=(num_train_samples, *im_shape))
    h5w.create_dataset('masks', shape=(num_train_samples, *im_shape[:-1]))
    h5w.create_dataset('metadata', data=np.concatenate([f['metadata'] for f in train_files]))
    current_len = 0
    for f in train_files:
        data_len = len(f['images'])
        h5w['images'][current_len:current_len + data_len] = f['images']
        h5w['masks'][current_len:current_len + data_len] = f['masks']
        current_len += data_len