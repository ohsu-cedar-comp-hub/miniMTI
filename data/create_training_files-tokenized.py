import os
import argparse
import h5py

def create_virtual_layout(files):
    num_samples = sum([f['tokens'].shape[0] for f in files])
    print(num_samples)
    ims_layout = h5py.VirtualLayout((num_samples, 288), dtype='int32')
    masks_layout = h5py.VirtualLayout((num_samples, 32,32), dtype='int32')
    meta_layout = h5py.VirtualLayout((num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
    
    current_len = 0
    for f in files:
        data_len = len(f['tokens'])
        print(data_len, f['tokens'].shape, f['masks'].shape, f['metadata'].shape)
        ims_layout[current_len:current_len + data_len] = h5py.VirtualSource(f['tokens'])
        masks_layout[current_len:current_len + data_len] = h5py.VirtualSource(f['masks'])
        meta_layout[current_len:current_len + data_len] = h5py.VirtualSource(f['metadata'])
        current_len += data_len
    return ims_layout, masks_layout, meta_layout
        

def create_file(fname, ims_layout, masks_layout, meta_layout):
    with h5py.File(fname,mode='w') as h5w:
        h5w.create_virtual_dataset('tokens', ims_layout)
        h5w.create_virtual_dataset('masks', masks_layout)
        h5w.create_virtual_dataset('metadata', meta_layout)
    

if __name__ == '__main__':
    train_dir = '/home/exacloud/gscratch/ChangLab/tokenized-crc-data-individual/train'
    val_dir = '/home/exacloud/gscratch/ChangLab/tokenized-crc-data-individual/val'
    train_files, val_files = [],[]
    for h5file in os.listdir(train_dir):
        if not h5file.endswith('.h5'): continue
        train_files.append(h5py.File(f"{train_dir}/{h5file}", "r"))
        
    for h5file in os.listdir(val_dir):
        train_files.append(h5py.File(f"{val_dir}/{h5file}", "r"))
            
    train_layouts = create_virtual_layout(train_files)
    val_layouts = create_virtual_layout(val_files)
    
    create_file(f'{train_dir}/train.h5', *train_layouts)
    create_file(f'{val_dir}/val.h5', *val_layouts)
            
    
    
    
