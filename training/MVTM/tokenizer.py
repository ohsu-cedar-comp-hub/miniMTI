import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange

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


class Tokenizer:
    def __init__(
        self, 
        if_config_path,
        if_ckpt_path,
        vq_dim,
        vq_f_dim,
        num_if_channels,
        num_codes,
        he_config_path=None,
        he_ckpt_path=None,
        device=None
    ):
        self.num_codes = num_codes
        self.vq_dim = vq_dim
        self.vq_f_dim = vq_f_dim
        self.num_if_channels = num_if_channels
        if device is None: device = 'cpu'
        
        self.max_positions = vq_dim**2
        if_config = load_config(if_config_path)
        self.if_tokenizer = load_vqgan(if_config, if_ckpt_path).to(device)
        
        if he_config_path is not None:
            he_config = load_config(he_config_path)
            self.he_tokenizer = load_vqgan(he_config, he_ckpt_path).to(device)
            self.if_only = False
        else:
            self.if_only = True
            
    def _tokenize(self, x, tokenizer):
        _, _, [_, _, indices] = tokenizer.encode(x)
        return indices
    
    def tokenize(self, x):
        batch_size = x.shape[0]  # Get the batch size from input
        
        if self.if_only: 
            x = rearrange(x, 'b c h w -> (b c) 1 h w')
            tokens = self._tokenize(x, self.tokenizer)
            tokens = tokens.reshape(batch_size, self.num_if_channels * self.max_positions)
            
        else:
            x_if = x[:,:-3,:,:] #splice first N-3 channels
            x_if = rearrange(x_if, 'b c h w -> (b c) 1 h w') #rearrange to tensor of single channel images
            tokens_if = self._tokenize(x_if, self.if_tokenizer) #tokenize
            tokens_if = tokens_if.reshape(batch_size, self.num_if_channels * self.max_positions) # reshape to one sequence per im
            
            x_he = x[:,-3:,:,:] #splice last 3 channels
            tokens_he = self._tokenize(x_he, self.he_tokenizer) #tokenize
            tokens_he = tokens_he.reshape(batch_size, self.max_positions) # reshape to one sequence per im
            tokens_he += self.num_codes #append H&E vocab to IF vocab
          
            tokens = torch.cat([tokens_if, tokens_he], axis=1) #concatenate IF and H&E token sequences
            
        tokens += 3 #(BERT reserves tokens 0,1, and 2 for special tokens)
        
        return tokens
    
    
    def _detokenize(self, indices, num_channels, batch_size, tokenizer):
        indices = torch.clamp(indices - 3, min=0, max=self.num_codes - 1)
        z = tokenizer.quantize.get_codebook_entry(
            indices,
            shape=(batch_size * num_channels, self.vq_dim, self.vq_dim, self.vq_f_dim)
        )
        z = z.reshape([batch_size * num_channels, self.vq_f_dim, self.vq_dim, self.vq_dim])
        return tokenizer.decode(z)
    
    def detokenize(self, indices, batch_size=None):
        if batch_size is None:
            batch_size = indices.shape[0]
            
        if self.if_only:
            decoded = self._detokenize(indices, self.num_if_channels, batch_size, self.if_tokenizer)
            decoded = rearrange(decoded, '(b c) 1 h w -> b c h w', c=self.num_if_channels)
        else:
            indices_if = indices[:,:-self.max_positions]
            decoded_if = self._detokenize(indices_if, self.num_if_channels, batch_size, self.if_tokenizer)
            decoded_if = rearrange(decoded_if, '(b c) 1 h w -> b c h w', c=self.num_if_channels)

            indices_he = indices[:,-self.max_positions:]
            indices_he -= self.num_codes
            decoded_he = self._detokenize(indices_he, 1, batch_size, self.he_tokenizer) #H&E is 1 "channel" or code sequence
            
            decoded = torch.cat([decoded_if, decoded_he], axis=1)
        
        return decoded
        
        
        
            
        
        