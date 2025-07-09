import os
import sys
import h5py
import torch
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange

sys.path.append('../../data/')
from data import SingleCellDataset

# Import VQGAN model
from taming.models.vqgan import VQModel

def load_config(config_path):
    print(f'Loading config from {config_path}')
    return OmegaConf.load(config_path)

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
        try:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Make sure the checkpoint path is correct and is a valid VQGAN checkpoint file.")
            raise
    return model.eval()

def tokenize(x, tokenizer, batch_size, vq_dim, num_channels, device="cuda"):
    """Tokenize input using VQGAN"""
    with torch.no_grad():
        x = x.to(device)
        if num_channels > 1:
            x = rearrange(x, 'b c h w -> (b c) 1 h w')
        tokens = tokenizer.encode(x)[2][2]  # Get indices
        if num_channels > 1:
            tokens = tokens.reshape(batch_size, num_channels * vq_dim * vq_dim)
        else:
            tokens = tokens.reshape(batch_size, vq_dim * vq_dim)
    return tokens.cpu()

def main():
    parser = argparse.ArgumentParser(description='Pretokenize data for MVTM training')
    parser.add_argument('--config-path-if', type=str, required=True, help='Path to VQGAN config file (IF)')
    parser.add_argument('--ckpt-path-if', type=str, required=True, help='Path to the VQGAN checkpoint file (IF)')
    parser.add_argument('--config-path-he', type=str, required=False, help='Path to VQGAN config file (H&E)')
    parser.add_argument('--ckpt-path-he', type=str, required=False, help='Path to the VQGAN checkpoint file (H&E)')
    parser.add_argument('--train-file', type=str, required=True, help='Path to the training dataset file')
    parser.add_argument('--val-file', type=str, required=True, help='Path to the validation dataset file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save tokenized data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for tokenization')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--remove-he', action='store_true', help="remove last three channels if H&E is stored with IF")
    parser.add_argument('--downscale', action='store_true', help="downscale images 2x")
    parser.add_argument('--deconvolve-he', action='store_true', help="deconvolve H&E")
    parser.add_argument('--codebook-size', type=int, default=256, help="number of VQ codes")
    parser.add_argument('--num-channels', type=int, required=True, help="number of channels per image")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizers
    config_if = load_config(args.config_path_if)
    tokenizer_if = load_vqgan(config_if, ckpt_path=args.ckpt_path_if)
    vq_dim = 4
    
    if args.ckpt_path_he is not None:
        config_he = load_config(args.config_path_he)
        tokenizer_he = load_vqgan(config_he, ckpt_path=args.ckpt_path_he)
        he_channels = 3
    else:
        tokenizer_he = None
        he_channels = 0
    
    # Count actual channels after processing
    if args.remove_he:
        if_channels = args.num_channels - 3 if args.num_channels > 3 else args.num_channels
        he_channels = 0
    else:
        if_channels = args.num_channels - 3 if args.num_channels > 3 else args.num_channels
    
    print(f"IF channels: {if_channels}, H&E channels: {he_channels}")
    
    # Move models to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_if = tokenizer_if.to(device)
    if tokenizer_he is not None:
        tokenizer_he = tokenizer_he.to(device)
    
    # Create datasets
    train_file = h5py.File(args.train_file, 'r')
    val_file = h5py.File(args.val_file, 'r')
    
    train_data = SingleCellDataset(
        train_file['images'], train_file['masks'], train_file['metadata'], 
        args.downscale, False, args.remove_he, args.deconvolve_he, True
    )
    
    val_data = SingleCellDataset(
        val_file['images'], val_file['masks'], val_file['metadata'], 
        args.downscale, False, args.remove_he, args.deconvolve_he, True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create output files
    train_output = h5py.File(os.path.join(args.output_dir, 'train_tokenized.h5'), 'w')
    val_output = h5py.File(os.path.join(args.output_dir, 'val_tokenized.h5'), 'w')
    
    num_codes = args.codebook_size
    
    # Create token datasets with dimensions based on batch size
    train_tokens_shape = (len(train_data), (if_channels + (1 if he_channels > 0 else 0)) * vq_dim * vq_dim)
    val_tokens_shape = (len(val_data), (if_channels + (1 if he_channels > 0 else 0)) * vq_dim * vq_dim)
    
    train_tokens_dataset = train_output.create_dataset('tokens', train_tokens_shape, dtype='i4')
    val_tokens_dataset = val_output.create_dataset('tokens', val_tokens_shape, dtype='i4')
    
    # Copy masks and metadata
    train_masks = train_output.create_dataset('masks', data=train_file['masks'])
    val_masks = val_output.create_dataset('masks', data=val_file['masks'])
    
    train_metadata = train_output.create_dataset('metadata', data=train_file['metadata'])
    val_metadata = val_output.create_dataset('metadata', data=val_file['metadata'])
    
    # Save original images (optional, but can be useful for debugging)
    train_images = train_output.create_dataset('images', data=train_file['images'])
    val_images = val_output.create_dataset('images', data=val_file['images'])
    
    # Process training data
    print("Tokenizing training data...")
    idx = 0
    for batch in tqdm(train_loader):
        images, masks, meta = batch
        batch_size = images.shape[0]
        
        if args.remove_he or he_channels == 0:
            # Process IF only
            if_tokens = tokenize(images, tokenizer_if, batch_size, vq_dim, if_channels, device)
            tokens = if_tokens
        else:
            # Process IF and H&E separately
            if_images = images[:, :-3]
            he_images = images[:, -3:]
            
            if_tokens = tokenize(if_images, tokenizer_if, batch_size, vq_dim, if_channels, device)
            he_tokens = tokenize(he_images, tokenizer_he, batch_size, vq_dim, 1, device)
            
            # Add offset to H&E tokens (num_codes)
            he_tokens += num_codes
            
            # Concatenate tokens
            tokens = torch.cat([if_tokens, he_tokens], dim=1)
        
        # Add offset of 3 for BERT special tokens
        tokens += 3
        
        # Save tokens
        end_idx = min(idx + batch_size, len(train_data))
        train_tokens_dataset[idx:end_idx] = tokens[:end_idx-idx]
        idx = end_idx
    
    # Process validation data
    print("Tokenizing validation data...")
    idx = 0
    for batch in tqdm(val_loader):
        images, masks, meta = batch
        batch_size = images.shape[0]
        
        if args.remove_he or he_channels == 0:
            # Process IF only
            if_tokens = tokenize(images, tokenizer_if, batch_size, vq_dim, if_channels, device)
            tokens = if_tokens
        else:
            # Process IF and H&E separately
            if_images = images[:, :-3]
            he_images = images[:, -3:]
            
            if_tokens = tokenize(if_images, tokenizer_if, batch_size, vq_dim, if_channels, device)
            he_tokens = tokenize(he_images, tokenizer_he, batch_size, vq_dim, 1, device)
            
            # Add offset to H&E tokens
            he_tokens += num_codes
            
            # Concatenate tokens
            tokens = torch.cat([if_tokens, he_tokens], dim=1)
        
        # Add offset of 3 for BERT special tokens
        tokens += 3
        
        # Save tokens
        end_idx = min(idx + batch_size, len(val_data))
        val_tokens_dataset[idx:end_idx] = tokens[:end_idx-idx]
        idx = end_idx
    
    # Save configuration metadata
    train_output.attrs['vq_dim'] = vq_dim
    train_output.attrs['if_channels'] = if_channels
    train_output.attrs['he_channels'] = 1 if he_channels > 0 else 0
    train_output.attrs['codebook_size'] = num_codes
    
    val_output.attrs['vq_dim'] = vq_dim
    val_output.attrs['if_channels'] = if_channels
    val_output.attrs['he_channels'] = 1 if he_channels > 0 else 0
    val_output.attrs['codebook_size'] = num_codes
    
    # Close files
    train_file.close()
    val_file.close()
    train_output.close()
    val_output.close()
    
    print(f"Tokenization complete. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()