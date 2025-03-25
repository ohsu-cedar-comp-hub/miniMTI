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

def get_unmask_schedule(num_tokens, T, schedule='cosine'):
    if schedule == 'cosine':
        scheduler = cosine_schedule_masking_ratio
    elif schedule == 'linear':
        scheduler = lambda u:u
        
    total_unmasked = 0
    unmask_steps = []
    for t in range(T):
        unmask_ratio = 1 - scheduler(u= (T - (t + 1))/T)
        num_to_unmask = int(unmask_ratio * num_tokens) - total_unmasked
        total_unmasked += num_to_unmask
        unmask_steps.append(num_to_unmask)
        
    return unmask_steps
    
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
        config_path_if,    
        ckpt_path_if,     
        vq_dim,        
        vq_f_dim,
        config_path_he=None,
        ckpt_path_he=None,
        full_channel_mask=False
    ):
        super().__init__() 
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_channels = num_channels
        self.full_channel_mask = full_channel_mask
        self.vq_dim = vq_dim         
        self.vq_f_dim = vq_f_dim 
        
        #if no H&E VQGAN checkpoint is provided, then IF only
        self.if_only = True
        if ckpt_path_he is not None:
            self.if_only = False
        
        #When we quantize H&E, it is input as input as one channel,
        #but is quantized to just one token sequence (as opposed to 3),
        #so if H&E is included, we have to differentiate between the
        #number of input channels, and the number of input markers
        #(H&E is 3 channels, but really only one marker)
        if self.if_only:
            self.num_markers = self.num_channels
            self.num_if_channels = self.num_channels
        else:
            self.num_markers = self.num_channels - 3 + 1
            self.num_if_channels = self.num_channels - 3   

        # Load VQGAN model(s)
        config_if = load_config(config_path_if)
        self.tokenizer = load_vqgan(config_if, ckpt_path=ckpt_path_if)
        self.num_if_codes = num_codes 
        
        #load seperate VQGAN for H&E images
        if not self.if_only:
            config_he = load_config(config_path_he)
            self.tokenizer_he = load_vqgan(config_he, ckpt_path=ckpt_path_he)
            #assuming for now that H&E and IF have the same codebook sizes,
            #we want to append the H&E token vocab to the IF token vocab
            self.num_codes = num_codes * 2 
                                
        else:
            self.num_codes = num_codes
        #add the MASK token as the last token ID
        #(BERT reserves tokens 0,1, and 2 for special tokens)
        self.mask_id = num_codes + 3 

        # Ensure max_position_embeddings is sufficient for all position IDs
        self.max_positions = self.vq_dim ** 2
        config = RobertaConfig(
            vocab_size=self.num_codes + 4,
            type_vocab_size=self.num_markers,
            max_position_embeddings=self.max_positions,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=latent_dim * 4
        )
        print(f'vq_dim: {self.vq_dim}')
        print(f'vq_f_dim: {self.vq_f_dim}')
        print(f'IF config_path: {config_path_if}')
        print(f'IF ckpt_path: {ckpt_path_if}')
        
        self.mvtm = RobertaForMaskedLM(config)

        
    def _tokenize(self, x, tokenizer):
        '''returns codebook indices from provided VQGAN model'''
        _, _, [_, _, indices] = tokenizer.encode(x)
        return indices
    
    
    def tokenize(self, x, batch_size):
        if self.if_only: 
            x = rearrange(x, 'b c h w -> (b c) 1 h w')
            tokens = self._tokenize(x, self.tokenizer)
            tokens = tokens.reshape(batch_size, self.num_if_channels * self.max_positions)
            
        else:
            x_if = x[:,:-3,:,:] #splice first N-3 channels
            x_if = rearrange(x_if, 'b c h w -> (b c) 1 h w') #rearrange to tensor of single channel images
            tokens_if = self._tokenize(x_if, self.tokenizer) #tokenize
            tokens_if = tokens_if.reshape(batch_size, self.num_if_channels * self.max_positions) # reshape to one sequence per im
            
            x_he = x[:,-3:,:,:] #splice last 3 channels
            tokens_he = self._tokenize(x_he, self.tokenizer_he) #tokenize
            tokens_he = tokens_he.reshape(batch_size, self.max_positions) # reshape to one sequence per im
            tokens_he += self.num_if_codes #append H&E vocab to IF vocab
          
            tokens = torch.cat([tokens_if, tokens_he], axis=1) #concatenate IF and H&E token sequences
            
        tokens += 3 #(BERT reserves tokens 0,1, and 2 for special tokens)
        
        return tokens
    
    
    def _detokenize(self, indices, num_channels, batch_size, tokenizer):
        indices = torch.clamp(indices - 3, min=0, max=self.num_if_codes - 1)
        z = tokenizer.quantize.get_codebook_entry(
            indices,
            shape=(batch_size * num_channels, self.vq_dim, self.vq_dim, self.vq_f_dim)
        )
        z = z.reshape([batch_size * num_channels, self.vq_f_dim, self.vq_dim, self.vq_dim])
        return tokenizer.decode(z)
    
    def detokenize(self, indices, batch_size):
        if self.if_only:
            decoded = self._detokenize(indices, self.num_if_channels, batch_size, self.tokenizer)
            decoded = rearrange(decoded, '(b c) 1 h w -> b c h w', c=self.num_if_channels)
        else:
            indices_if = indices[:,:-self.max_positions]
            decoded_if = self._detokenize(indices_if, self.num_if_channels, batch_size, self.tokenizer)
            decoded_if = rearrange(decoded_if, '(b c) 1 h w -> b c h w', c=self.num_if_channels)

            indices_he = indices[:,-self.max_positions:]
            indices_he -= self.num_if_codes
            decoded_he = self._detokenize(indices_he, 1, batch_size, self.tokenizer_he) #H&E is 1 "channel" or code sequence
            
            decoded = torch.cat([decoded_if, decoded_he], axis=1)
        
        return decoded
        
    def mask_channels(self, token_ids, masked_ch_idx=None):
        if masked_ch_idx is not None:
            labels = token_ids.clone()
            for channel_i in masked_ch_idx:
                token_ids[:, self.max_positions * channel_i:self.max_positions * (channel_i + 1)] = self.mask_id
            labels[token_ids != self.mask_id] = -100
        elif self.full_channel_mask:
            seq_length = self.max_positions * self.num_markers
            num_masked = int(cosine_schedule_masking_ratio() * (self.num_markers - 1)) + 1
            assert num_masked > 0 and num_masked < self.num_markers
            rand_indices = torch.rand(token_ids.shape[0], self.num_markers, device=self.device).argsort(dim=-1)
            masked_indices = rand_indices[:, :num_masked]
            mask = torch.zeros((token_ids.shape[0], seq_length), dtype=torch.bool, device=self.device)
            start_positions = masked_indices * self.max_positions
            offsets = torch.arange(self.max_positions, device=self.device).unsqueeze(0).unsqueeze(0)
            expanded_start_positions = start_positions.unsqueeze(-1) + offsets
            expanded_start_positions = expanded_start_positions.view(token_ids.shape[0], -1)
            mask.scatter_(1, expanded_start_positions, True)
            labels = token_ids.clone()
            labels[~mask] = -100
            token_ids[mask] = self.mask_id
        else:
            seq_length = self.max_positions * self.num_markers
            batch_size = token_ids.shape[0]
            num_masked = int(cosine_schedule_masking_ratio() * (seq_length - 1)) + 1
            rand_indices = torch.stack([torch.randperm(seq_length, device=self.device) for _ in range(batch_size)])
            masked_indices = rand_indices[:, :num_masked]
            mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=self.device)
            mask.scatter_(1, masked_indices, True)
            labels = token_ids.clone()
            labels[~mask] = -100
            token_ids[mask] = self.mask_id
        return token_ids, labels
        
    def unmask(self, tokens, labels, logits, batch_size, k, temp=1.0):
        # Find all mask locations
        mask_locations = torch.where(labels != -100)
        # If there are no masks, return the original tokens
        if len(mask_locations[0]) == 0:
            return tokens, None
        # Apply temperature to logits if temperature is not 0
        if temp == 0:
            scores = F.softmax(logits, dim=2)
        else:
            scaled_logits = logits / temp
            scores = F.softmax(scaled_logits, dim=2)
        # Create a mask tensor
        mask = (labels != -100).float()
        # Calculate the max probability for each position
        max_probs, _ = (scores * mask.unsqueeze(-1)).max(dim=2)
        # Find the positions of the K most probable masks for each batch item
        K = int(min(k, mask.sum(dim=1).min().item()))  # Ensure K is not larger than the minimum number of masks in any batch item
        top_k_probs, top_k_positions = torch.topk(max_probs, k=K, dim=1)
        # Create indices for gathering
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, K)
        if temp == 0:
            # Use greedy decoding (argmax) when temperature is 0
            selected_tokens = logits[batch_indices, top_k_positions].argmax(dim=2)
        else:
            # Sample from the distribution for non-zero temperatures
            # Reshape the scores for multinomial sampling
            flattened_scores = scores[batch_indices, top_k_positions].view(-1, scores.size(-1))
            sampled_indices = torch.multinomial(flattened_scores, num_samples=1).view(batch_size, K)
            selected_tokens = sampled_indices
        # Update tokens with selected values
        tokens[batch_indices, top_k_positions] = selected_tokens
        labels[batch_indices, top_k_positions] = -100
        return tokens, labels
    
    def forward(self, x, masked_ch_idx=None):
        batch_size = x.shape[0]
        device = x.device
        with torch.no_grad():
            token_ids = self.tokenize(x, batch_size)
        
        #input_ids, labels = self.mask_channels(token_ids.clone(), masked_ch_idx=masked_ch_idx)
        input_ids, labels = self.mask_channels(token_ids.clone())
        
        print(f"{input_ids.shape=}")
        print(f"{input_ids.max=}")
        print(f"{input_ids.min=}")
        
        type_ids = torch.cat([torch.ones(self.max_positions, device=device) * i for i in range(self.num_markers)]).long()
        position_ids = torch.cat([torch.arange(self.max_positions, device=device) for _ in range(self.num_markers)]).long()
        
        print(f"{type_ids.shape=}")
        print(f"{type_ids.min=}")
        print(f"{type_ids.max=}")
        print(f"{position_ids.shape=}")
        print(f"{position_ids.min=}")
        print(f"{position_ids.max=}")

        out = self.mvtm(input_ids=input_ids, token_type_ids=type_ids, position_ids=position_ids, labels=labels)
        
        return out, token_ids, labels
    
    def predict(self, x, masked_ch_idx=None, T=10, temp=1.0, schedule='cosine', output_attentions=False, output_hidden_states=False):
        batch_size = x.shape[0]
        device = x.device
        with torch.no_grad():
            token_ids = self.tokenize(x, batch_size)
        
        if masked_ch_idx is None:
            input_ids = token_ids.clone()
            labels = torch.ones(input_ids.shape, device=device) * -100
        input_ids, labels = self.mask_channels(token_ids.clone(), masked_ch_idx=masked_ch_idx)
        
        type_ids = torch.cat([torch.ones(self.max_positions, device=device)*i for i in range(self.num_markers)]).long()
        position_ids = torch.cat([torch.arange(self.max_positions, device=device) for _ in range(self.num_markers)]).long()
        
        if masked_ch_idx is None:
            unmask_schedule = [1]
        else:
            unmask_schedule = get_unmask_schedule(self.max_positions * len(masked_ch_idx), T, schedule=schedule)
        for k in unmask_schedule:
            if k == 0: continue
            out = self.mvtm(input_ids=input_ids, token_type_ids=type_ids, position_ids=position_ids, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            print(out['loss'])
            if (labels == -100).all(): break
            input_ids, labels = self.unmask(input_ids, labels, out['logits'], batch_size, k, temp=temp)
        return out, input_ids, labels
                       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        def linear_lr_schedule(epoch):
            return 1 - epoch / self.trainer.max_epochs
        scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_schedule)
        return [optimizer], [scheduler]
        
    def training_step(self, train_batch, batch_idx):
        gt, mask, meta = train_batch
        out, _, _ = self(gt)
        self.log('loss', out['loss'], sync_dist=True)
        return out['loss']
        
    def validation_step(self, val_batch, val_idx):
        gt, mask, meta = val_batch
        out, token_ids, labels = self(gt)
        self.log('val_loss', out['loss'], sync_dist=True)
