import torch
import math
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaForMaskedLM, RobertaConfig
from einops import repeat, rearrange
from omegaconf import OmegaConf
import argparse
print('before taming models imported')

from taming.models.vqgan import VQModel

print('taming models imported')

def load_config(config_path):
    print('Enter config path')
    return OmegaConf.load(config_path)

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()
    
def cosine_schedule_masking_ratio(u=None):
    """
    Returns a masking ratio sampled from a truncated arccos distribution.
    
    The density function is p(r) = (2/π) * (1 - r^2)^(-1/2), where r ∈ [0, 1].
    This distribution has an expected masking rate of 0.64, with a bias towards higher masking rates.
    
    Returns:
        float: A sampled masking ratio between 0 and 1.
    """
    # Sample a uniform value between 0 and 1
    if u is None:
        u = torch.rand(1).item()
    
    # Apply the inverse CDF of the truncated arccos distribution
    r = math.sin(math.pi * u / 2)
    
    return r

def get_unmask_schedule(num_tokens, T):
    total_unmasked = 0
    unmask_steps = []
    for t in range(T):
        mr = 1 - cosine_schedule_masking_ratio(u=(t+1)/T)
        num_masked = int(mr * num_tokens)
        num_unmasked = num_tokens - num_masked
        num_to_unmask = num_unmasked - total_unmasked
        unmask_steps.append(num_to_unmask)
        total_unmasked += num_to_unmask
    return list(reversed(unmask_steps))
    
class IF_MVTM(pl.LightningModule):
    def __init__(
        self,
        lr,
        weight_decay,
        num_channels,
        num_layers,
        num_heads,
        latent_dim,
        num_codes,
        config_path,   # added config_path argument
        ckpt_path,     # added ckpt_path argument
        vq_dim,        # added vq_dim argument
        vq_f_dim       # added vq_f_dim argument
    ):
        super().__init__() 
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_channels = num_channels

        self.vq_dim = vq_dim         # use passed vq_dim argument
        self.vq_f_dim = vq_f_dim     # use passed vq_f_dim argument

        # Load VQGAN model
        config_aced = load_config(config_path)   # use passed config_path argument
        model_aced = load_vqgan(config_aced, ckpt_path=ckpt_path)  # use passed ckpt_path argument

        self.tokenizer = model_aced

        self.mask_id = num_codes + 3

        config = RobertaConfig(
            vocab_size=num_codes + 4,
            type_vocab_size=num_channels,
            max_position_embeddings=self.vq_dim ** 2,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=latent_dim * 4
        )
        print(f'vq_dim: {self.vq_dim}')
        print(f'vq_f_dim: {self.vq_f_dim}')
        print(f'config_path: {config_path}')
        print(f'ckpt_path: {ckpt_path}')
        
        self.mvtm = RobertaForMaskedLM(config)


        
    def tokenize(self, x):
        z, _, [_, _, indices] = self.tokenizer.encode(x)
        return indices + 3
    
    def detokenize(self, indices, batch_size):
        indices = torch.clamp(indices - 3, min=0)
        z = self.tokenizer.quantize.get_codebook_entry(
            indices,
            shape=(batch_size * self.num_channels, self.vq_dim, self.vq_dim, self.vq_f_dim)
        )
        z = z.reshape([batch_size * self.num_channels, self.vq_f_dim, self.vq_dim, self.vq_dim])
        return self.tokenizer.decode(z)
    
    def decode(self, tokens, labels, logits, batch_size):
        mask_locations = torch.where(labels != -100)
        scores = F.softmax(logits, dim=2)
        tokens[mask_locations] = logits[mask_locations].argmax(dim=1)
        output = rearrange(tokens, 'b (c h w) -> (b c) h w', c=self.num_channels, h=self.vq_dim)
        detok = self.detokenize(output, batch_size)
        detok = rearrange(detok, '(b c) 1 h w -> b c h w', c=self.num_channels)
        return detok
        
    
    def unmask(self, tokens, labels, logits, batch_size, k):
        # Find all mask locations
        mask_locations = torch.where(labels != -100)
        # If there are no masks, return the original tokens
        if len(mask_locations[0]) == 0:
            return tokens, None
        # Calculate softmax scores
        scores = F.softmax(logits, dim=2)
        # Create a mask tensor
        mask = (labels != -100).float()
        # Calculate the max probability for each position
        max_probs, _ = (scores * mask.unsqueeze(-1)).max(dim=2)
        # Find the positions of the K most probable masks for each batch item
        K = int(min(k, mask.sum(dim=1).min().item()))  # Ensure K is not larger than the minimum number of masks in any batch item
        top_k_probs, top_k_positions = torch.topk(max_probs, k=K, dim=1)
        # Create indices for gathering
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, K)
        # Decode the most probable masked token for each batch item
        tokens[batch_indices, top_k_positions] = logits[batch_indices, top_k_positions].argmax(dim=2)
        labels[batch_indices, top_k_positions] = -100
        return tokens, labels
    
    def forward(self, x, masked_ch_idx=None):
        batch_size = x.shape[0]
        device = x.device
        with torch.no_grad():
            x = rearrange(x, 'b c h w -> (b c) 1 h w')
            token_ids = self.tokenize(x)
            token_ids = token_ids.reshape(batch_size, self.num_channels * (self.vq_dim**2))
        
        input_ids, labels = self.mask_channels(token_ids.clone(), masked_ch_idx=masked_ch_idx)

        type_ids = torch.cat([torch.ones(self.vq_dim**2, device=device) * i for i in range(self.num_channels)]).long()
        position_ids = torch.cat([torch.arange(self.vq_dim**2, device=device) for _ in range(self.num_channels)]).long()

        out = self.mvtm(input_ids=input_ids, token_type_ids=type_ids, position_ids=position_ids, labels=labels)
        
        return out, token_ids, labels
    
    def predict(self, x, masked_ch_idx=None):
        batch_size = x.shape[0]
        device = x.device
        with torch.no_grad():
            x = rearrange(x, 'b c h w -> (b c) 1 h w')
            token_ids = self.tokenize(x)
            token_ids = token_ids.reshape(batch_size,self.num_channels * (self.vq_dim**2))
        
        input_ids, labels = self.mask_channels(token_ids.clone(), masked_ch_idx=masked_ch_idx)
        
        type_ids = torch.cat([torch.ones(self.vq_dim**2, device=device)*i for i in range(self.num_channels)]).long()
        position_ids = torch.cat([torch.arange(self.vq_dim**2, device=device) for _ in range(self.num_channels)]).long()
        
        T = 30
        unmask_schedule = get_unmask_schedule(self.vq_dim**2 * len(masked_ch_idx), T)
        for k in unmask_schedule:
            if k == 0: continue
            out = self.mvtm(input_ids=input_ids, token_type_ids=type_ids, position_ids=position_ids, labels=labels)
            input_ids, labels = self.unmask(input_ids, labels, out['logits'], batch_size, k)
        return out, input_ids, labels
                       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        def linear_lr_schedule(epoch):
            return 1 - epoch / self.trainer.max_epochs
        scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_schedule)
        return [optimizer], [scheduler]
  
    def mask_channels(self, token_ids, masked_ch_idx=None):
        if masked_ch_idx is not None:
            for channel_i in masked_ch_idx:
                token_ids[:, self.vq_dim**2 * channel_i:self.vq_dim**2 * (channel_i + 1)] = self.mask_id
            labels = token_ids.clone()
            labels[token_ids != self.mask_id] = -100
        else:
            seq_length = (self.vq_dim**2) * self.num_channels
            num_masked = int(cosine_schedule_masking_ratio() * ((self.vq_dim**2 * self.num_channels) - 1)) + 1
            rand_indices = torch.rand(token_ids.shape[0], self.num_channels, device=self.device).argsort(dim=-1)
            masked_indices = rand_indices[:, :num_masked]
            mask = torch.zeros((token_ids.shape[0], seq_length), dtype=torch.bool, device=self.device)
            start_positions = masked_indices * self.vq_dim**2
            offsets = torch.arange(self.vq_dim**2, device=self.device).unsqueeze(0).unsqueeze(0)
            expanded_start_positions = start_positions.unsqueeze(-1) + offsets
            expanded_start_positions = expanded_start_positions.view(token_ids.shape[0], -1)
            mask.scatter_(1, expanded_start_positions, True)
        
            labels = token_ids.clone()
            labels[~mask] = -100
            token_ids[mask] = self.mask_id

        return token_ids, labels
        
    def training_step(self, train_batch, batch_idx):
        gt, mask, meta = train_batch
        out, _, _ = self(gt)
        self.log('loss', out['loss'], sync_dist=True)
        return out['loss']
        
    def validation_step(self, val_batch, val_idx):
        gt, mask, meta = val_batch
        out, token_ids, labels = self(gt)
        self.log('val_loss', out['loss'], sync_dist=True)

