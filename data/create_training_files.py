import os
import argparse
import h5py

def create_virtual_layout(files):
    num_samples = sum([f['images'].shape[0] for f in files])
    im_shape = files[0]['images'].shape[1:]
    ims_layout = h5py.VirtualLayout((num_samples, *im_shape), dtype='uint8')
    masks_layout = h5py.VirtualLayout((num_samples, *im_shape[:-1]), dtype='uint8')
    meta_layout = h5py.VirtualLayout((num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
    
    current_len = 0
    for f in files:
            data_len = len(f['images'])
            ims_layout[current_len:current_len + data_len] = h5py.VirtualSource(f['images'])
            masks_layout[current_len:current_len + data_len] = h5py.VirtualSource(f['masks'])
            meta_layout[current_len:current_len + data_len] = h5py.VirtualSource(f['metadata'])
            current_len += data_len
    return ims_layout, masks_layout, meta_layout
        

def create_file(fname, ims_layout, masks_layout, meta_layout):
    with h5py.File(fname,mode='w') as h5w:
        h5w.create_virtual_dataset('images', ims_layout)
        h5w.create_virtual_dataset('masks', masks_layout)
        h5w.create_virtual_dataset('metadata', meta_layout)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merges individual h5 files per sample into combined training and validation files')
    parser.add_argument("--data-dir", type=str, required=True, help="Path to directory containing h5 files")
    parser.add_argument("--val-samples", type=str, default=None, help="path to file containing validation sample names")
    parser.add_argument("--train-samples", type=str, required=False, default=None, help="path to file containing training sample names")
    parser.add_argument("--batch-name", type=str, default=None, help="Name for validation batch")
    args = parser.parse_args()
    
    with open(args.val_samples) as f:
        val_samples = [s.strip() for s in f.readlines()]
        
    if args.train_samples is not None:
        with open(args.train_samples) as f:
            train_samples = [s.strip() for s in f.readlines()]
        
    train_files, val_files = [],[]
    for h5file in os.listdir(args.data_dir):
        if not h5file.endswith('h5'): continue
        if all([s not in h5file for s in val_samples]) and ('train' not in h5file) and ('panel_select' not in h5file) and ('val' not in h5file):
            if args.train_samples is not None:
                if any([s in h5file for s in train_samples]):
                    train_files.append(h5py.File(f"{args.data_dir}/{h5file}", "r"))
            else:
                train_files.append(h5py.File(f"{args.data_dir}/{h5file}", "r"))
        else:
            val_files.append(h5py.File(f"{args.data_dir}/{h5file}", "r"))
            
    train_layouts = create_virtual_layout(train_files)
    val_layouts = create_virtual_layout(val_files)
    
    create_file(f'{args.data_dir}/train-{args.batch_name}.h5', *train_layouts)
    create_file(f'{args.data_dir}/val-{args.batch_name}.h5', *val_layouts)
            
    
    
    
