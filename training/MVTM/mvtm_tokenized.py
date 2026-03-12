import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import gc
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaForMaskedLM, RobertaConfig
from einops import repeat, rearrange
import weakref

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
    
class Tokenized_MVTM(pl.LightningModule):
    def __init__(
        self,
        num_markers,
        num_layers,
        num_heads,
        latent_dim,
        num_codes,
        vq_dim,
        lr=None,
        weight_decay=None,
        full_channel_mask=False,
        if_only_mask=False
    ):
        super().__init__() 
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_markers = num_markers
        self.full_channel_mask = full_channel_mask
        self.if_only_mask = if_only_mask
        self.vq_dim = vq_dim  
        
        self.max_positions = self.vq_dim ** 2
        
        self.num_codes = num_codes * 2 #concacentate H&E vocab to IF vocab
        
        # Add the MASK token as the last token ID
        # (BERT reserves tokens 0,1, and 2 for special tokens)
        self.mask_id = num_codes + 3 
        

        # Ensure max_position_embeddings is sufficient for all position IDs
        config = RobertaConfig(
            vocab_size=self.num_codes + 4,
            type_vocab_size=self.num_markers,
            max_position_embeddings=self.max_positions,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=latent_dim * 4
        )
        
        self.mvtm = RobertaForMaskedLM(config)
        
    def extract_channel_tokens(self, token_ids, channel_indices):
        """Extract tokens for specific channels from the full token sequence.

        Args:
            token_ids: [batch, num_markers * max_positions]
            channel_indices: List of channel indices to extract (in order they should appear)

        Returns:
            extracted_tokens: [batch, len(channel_indices) * max_positions]
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Extract tokens for each channel
        extracted = []
        for ch_idx in channel_indices:
            start_idx = self.max_positions * ch_idx
            end_idx = self.max_positions * (ch_idx + 1)
            extracted.append(token_ids[:, start_idx:end_idx])

        return torch.cat(extracted, dim=1)

    def mask_channels(self, token_ids, masked_ch_idx=None, input_ch_idx=None, output_ch_idx=None):
        device = token_ids.device
        batch_size = token_ids.shape[0]

        # New interface: input_ch_idx and output_ch_idx
        if input_ch_idx is not None and output_ch_idx is not None:
            # Extract only input + output channel tokens
            all_ch_idx = list(input_ch_idx) + list(output_ch_idx)
            token_ids = self.extract_channel_tokens(token_ids, all_ch_idx)

            # Create labels
            labels = token_ids.clone()

            # Mask only the output channels (which are at the end of the sequence)
            num_input_channels = len(input_ch_idx)
            for i, channel_i in enumerate(output_ch_idx):
                # Channels are now indexed relative to the extracted sequence
                relative_idx = num_input_channels + i
                start_idx = self.max_positions * relative_idx
                end_idx = self.max_positions * (relative_idx + 1)
                token_ids[:, start_idx:end_idx] = self.mask_id

            # Set labels: -100 for unmasked positions
            labels[token_ids != self.mask_id] = -100

            return token_ids, labels, all_ch_idx

        # Legacy interface: masked_ch_idx
        elif masked_ch_idx is not None:
            # Specific channel masking - more efficient implementation
            labels = token_ids.clone()
            # Process in place with less memory allocation
            for channel_i in masked_ch_idx:
                start_idx = self.max_positions * channel_i
                end_idx = self.max_positions * (channel_i + 1)
                token_ids[:, start_idx:end_idx] = self.mask_id
            labels[token_ids != self.mask_id] = -100
        elif hasattr(self, 'if_only_mask') and self.if_only_mask:
            # Mask only IF channel tokens (first 17 channels), preserve H&E tokens (last channel)
            if_seq_length = self.max_positions * (self.num_markers - 1)  # 17 IF channels
            he_seq_length = self.max_positions  # 1 H&E channel
            
            # Calculate number of IF tokens to mask (random number from the first 17*16 tokens)
            num_masked = int(cosine_schedule_masking_ratio() * (if_seq_length - 1)) + 1
            assert num_masked > 0 and num_masked < if_seq_length
            
            # Create labels before modifying token_ids to avoid extra clone
            labels = token_ids.clone()
            
            with torch.no_grad():
                # Process in batches to reduce memory usage
                batch_size_chunk = 8  # Process 8 examples at a time
                for i in range(0, batch_size, batch_size_chunk):
                    end_idx = min(i + batch_size_chunk, batch_size)
                    
                    # Create mask just for the IF tokens in this chunk
                    chunk_mask = torch.zeros((end_idx - i, if_seq_length), dtype=torch.bool, device=device)
                    
                    # Process each example in the chunk
                    for j in range(i, end_idx):
                        rand_indices = torch.randperm(if_seq_length, device=device)[:num_masked]
                        chunk_mask[j-i].scatter_(0, rand_indices, True)
                        del rand_indices  # Immediately free memory
                    
                    # Apply masks for this chunk (only to IF tokens)
                    labels[i:end_idx, :if_seq_length][~chunk_mask] = -100
                    token_ids[i:end_idx, :if_seq_length][chunk_mask] = self.mask_id
                    
                    # H&E tokens (last 16 tokens) are never masked, so set their labels to -100
                    labels[i:end_idx, if_seq_length:] = -100
                    
                    del chunk_mask  # Free memory immediately
        elif self.full_channel_mask:
            # Full channel masking
            seq_length = self.max_positions * self.num_markers
            
            # Calculate number of channels to mask
            num_masked = int(cosine_schedule_masking_ratio() * (self.num_markers - 1)) + 1
            assert num_masked > 0 and num_masked < self.num_markers
            
            # Use a more memory-efficient approach for cached offsets
            if not hasattr(self, '_cached_offsets_ref'):
                # Store a weak reference to avoid memory leaks
                offsets = torch.arange(self.max_positions, device=device).unsqueeze(0).unsqueeze(0)
                self._cached_offsets_ref = weakref.ref(offsets)
                self._cached_offsets_device = device
            elif not hasattr(self, '_cached_offsets_device') or self._cached_offsets_device != device:
                # Recreate on new device
                offsets = torch.arange(self.max_positions, device=device).unsqueeze(0).unsqueeze(0)
                self._cached_offsets_ref = weakref.ref(offsets)
                self._cached_offsets_device = device
            
            offsets = self._cached_offsets_ref()
            if offsets is None:
                # Recreate if it was garbage collected
                offsets = torch.arange(self.max_positions, device=device).unsqueeze(0).unsqueeze(0)
                self._cached_offsets_ref = weakref.ref(offsets)
            
            # Generate random indices more efficiently
            with torch.no_grad():
                # Generate indices in blocks to save memory
                rand_indices = torch.rand(batch_size, self.num_markers, device=device).argsort(dim=-1)
                masked_indices = rand_indices[:, :num_masked]
                del rand_indices  # Immediately release memory
                
                # Create mask for the tokens to be masked
                mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
                start_positions = masked_indices * self.max_positions
                
                # Process in smaller batches to reduce peak memory
                batch_size_chunk = 16
                for i in range(0, batch_size, batch_size_chunk):
                    end_idx = min(i + batch_size_chunk, batch_size)
                    chunk_size = end_idx - i
                    
                    # Process this chunk
                    expanded = start_positions[i:end_idx].unsqueeze(-1) + offsets
                    expanded = expanded.reshape(chunk_size, -1)
                    mask[i:end_idx].scatter_(1, expanded, True)
                    
                    # Clear intermediate tensors
                    del expanded
                
                del start_positions, masked_indices  # Release memory
                
                # Create copy of token_ids for labels
                labels = token_ids.clone()
                labels[~mask] = -100
                token_ids[mask] = self.mask_id
                
                del mask  # Release memory
        else:
            # Random token masking
            seq_length = self.max_positions * self.num_markers
            num_masked = int(cosine_schedule_masking_ratio() * (seq_length - 1)) + 1
            
            # Create labels before modifying token_ids to avoid extra clone
            labels = token_ids.clone()
            
            with torch.no_grad():
                # Process in batches to reduce memory usage
                batch_size_chunk = 8  # Process 8 examples at a time
                for i in range(0, batch_size, batch_size_chunk):
                    end_idx = min(i + batch_size_chunk, batch_size)
                    
                    # Create mask just for this chunk
                    chunk_mask = torch.zeros((end_idx - i, seq_length), dtype=torch.bool, device=device)
                    
                    # Process each example in the chunk
                    for j in range(i, end_idx):
                        rand_indices = torch.randperm(seq_length, device=device)[:num_masked]
                        chunk_mask[j-i].scatter_(0, rand_indices, True)
                        del rand_indices  # Immediately free memory
                    
                    # Apply masks for this chunk
                    labels[i:end_idx][~chunk_mask] = -100
                    token_ids[i:end_idx][chunk_mask] = self.mask_id
                    
                    del chunk_mask  # Free memory immediately
        
        # Force CUDA to synchronize to release memory
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Return None for all_ch_idx in legacy mode (using all channels)
        return token_ids, labels, None
        
    def unmask(self, tokens, labels, logits, batch_size, k, temp=1.0):
        device = tokens.device
        
        with torch.no_grad():
            # Find all mask locations
            mask = (labels != -100)
            # If there are no masks, return the original tokens
            if not mask.any():
                return tokens, None
                
            # Apply temperature to logits but only where needed to save memory
            # Process in batch chunks to reduce memory usage
            max_probs = torch.zeros_like(labels, dtype=torch.float32)
            max_probs.masked_fill_(~mask, float('-inf'))
            
            # Process in small batches to reduce memory usage
            chunk_size = 8
            for b_start in range(0, batch_size, chunk_size):
                b_end = min(b_start + chunk_size, batch_size)
                
                for b in range(b_start, b_end):
                    # Get only the positions where we have masks for this batch item
                    mask_positions = mask[b].nonzero(as_tuple=True)[0]
                    
                    if len(mask_positions) == 0:
                        continue
                    
                    # Process only the masked positions to save memory
                    for pos_idx, pos in enumerate(mask_positions):
                        if temp == 0:
                            max_probs[b, pos] = logits[b, pos].max().item()
                        else:
                            # Apply softmax only on this position
                            probs = F.softmax(logits[b, pos] / temp, dim=0)
                            max_probs[b, pos] = probs.max().item()
            
            # Find the positions of the K most probable masks for each batch item
            # Process in batch chunks to reduce memory
            for b_start in range(0, batch_size, chunk_size):
                b_end = min(b_start + chunk_size, batch_size)
                
                for b in range(b_start, b_end):
                    # Find minimum number of masks for this batch
                    num_masks = mask[b].sum().item()
                    if num_masks == 0:
                        continue
                    
                    # Use smaller k if needed
                    k_batch = min(k, num_masks)
                    
                    # Get top-k positions for this batch
                    top_k_probs, top_k_positions = torch.topk(max_probs[b], k=k_batch)
                    
                    # Sample or select tokens for each position
                    for i, pos in enumerate(top_k_positions):
                        if temp == 0:
                            # Greedy selection
                            tokens[b, pos] = logits[b, pos].argmax().item()
                        else:
                            # Sample from distribution
                            probs = F.softmax(logits[b, pos] / temp, dim=0)
                            tokens[b, pos] = torch.multinomial(probs, num_samples=1).item()
                    
                    # Mark positions as unmasked
                    labels[b, top_k_positions] = -100
                    
                    del top_k_probs, top_k_positions
            
            # Clean up
            del mask, max_probs
            
            # Force CUDA to synchronize to release memory
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
            return tokens, labels
    
    def _get_position_type_ids(self, device, channel_indices=None):
        """Get position and type IDs for the token sequence.

        Args:
            device: torch device
            channel_indices: Optional list of channel indices. If provided, creates IDs for only these channels.
                           If None, creates IDs for all channels (legacy behavior).

        Returns:
            type_ids: Token type IDs indicating which channel each token belongs to
            position_ids: Position IDs indicating spatial position within each channel
        """
        if channel_indices is None:
            # Legacy behavior: use all channels with caching
            # Use weak references for these cached tensors to prevent memory leaks
            if not hasattr(self, '_cached_type_ids_ref'):
                type_ids = torch.cat([torch.ones(self.max_positions, device=device) * i for i in range(self.num_markers)]).long()
                position_ids = torch.cat([torch.arange(self.max_positions, device=device) for _ in range(self.num_markers)]).long()
                self._cached_type_ids_ref = weakref.ref(type_ids)
                self._cached_position_ids_ref = weakref.ref(position_ids)
                self._cached_ids_device = device
            elif not hasattr(self, '_cached_ids_device') or self._cached_ids_device != device:
                # Recreate on new device
                type_ids = torch.cat([torch.ones(self.max_positions, device=device) * i for i in range(self.num_markers)]).long()
                position_ids = torch.cat([torch.arange(self.max_positions, device=device) for _ in range(self.num_markers)]).long()
                self._cached_type_ids_ref = weakref.ref(type_ids)
                self._cached_position_ids_ref = weakref.ref(position_ids)
                self._cached_ids_device = device

            # Get from weak references
            type_ids = self._cached_type_ids_ref()
            position_ids = self._cached_position_ids_ref()

            # Recreate if they were garbage collected
            if type_ids is None or position_ids is None:
                type_ids = torch.cat([torch.ones(self.max_positions, device=device) * i for i in range(self.num_markers)]).long()
                position_ids = torch.cat([torch.arange(self.max_positions, device=device) for _ in range(self.num_markers)]).long()
                self._cached_type_ids_ref = weakref.ref(type_ids)
                self._cached_position_ids_ref = weakref.ref(position_ids)
        else:
            # New behavior: create IDs for only the specified channels
            # Type IDs use the actual channel indices
            type_ids = torch.cat([torch.ones(self.max_positions, device=device) * ch_idx for ch_idx in channel_indices]).long()
            # Position IDs cycle through 0-255 for each channel
            position_ids = torch.cat([torch.arange(self.max_positions, device=device) for _ in channel_indices]).long()

        return type_ids, position_ids
        
    def clear_cache(self):
        """Clear any cached tensors to free memory"""
        attrs_to_clear = [
            '_cached_type_ids_ref', '_cached_position_ids_ref', 
            '_cached_offsets_ref', '_cached_ids_device', '_cached_offsets_device'
        ]
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def forward(self, token_ids, masked_ch_idx=None, input_ch_idx=None, output_ch_idx=None):
        device = token_ids.device

        # Use a context manager to automatically clean up tensors
        try:
            input_ids, labels, all_ch_idx = self.mask_channels(
                token_ids.clone(),
                masked_ch_idx=masked_ch_idx,
                input_ch_idx=input_ch_idx,
                output_ch_idx=output_ch_idx
            )
            type_ids, position_ids = self._get_position_type_ids(device, channel_indices=all_ch_idx)

            # Regular forward pass without label smoothing
            out = self.mvtm(
                input_ids=input_ids,
                token_type_ids=type_ids,
                position_ids=position_ids,
                labels=labels
            )

            return out, token_ids, labels
        except Exception as e:
            # Clean cache on error to prevent memory leaks
            self.clear_cache()
            raise e
    
    def predict(self, token_ids, masked_ch_idx=None, input_ch_idx=None, output_ch_idx=None, T=10, temp=1.0, schedule='cosine', output_attentions=False, output_hidden_states=False, return_logits=False):
        batch_size = token_ids.shape[0]
        device = token_ids.device

        try:
            if masked_ch_idx is None and input_ch_idx is None:
                input_ids = token_ids.clone()
                labels = torch.ones(input_ids.shape, device=device) * -100
                all_ch_idx = None
            else:
                input_ids, labels, all_ch_idx = self.mask_channels(
                    token_ids.clone(),
                    masked_ch_idx=masked_ch_idx,
                    input_ch_idx=input_ch_idx,
                    output_ch_idx=output_ch_idx
                )

            type_ids, position_ids = self._get_position_type_ids(device, channel_indices=all_ch_idx)

            # Determine unmask schedule based on which parameters were provided
            if input_ch_idx is not None and output_ch_idx is not None:
                # New interface: unmask based on output channels
                num_masked_tokens = self.max_positions * len(output_ch_idx)
            elif masked_ch_idx is not None:
                # Legacy interface: unmask based on masked channels
                num_masked_tokens = self.max_positions * len(masked_ch_idx)
            else:
                # No masking
                num_masked_tokens = 1

            unmask_schedule = get_unmask_schedule(num_masked_tokens, T, schedule=schedule)

            for k in unmask_schedule:
                if k == 0: continue

                # Free memory before each forward pass
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                out = self.mvtm(
                    input_ids=input_ids,
                    token_type_ids=type_ids,
                    position_ids=position_ids,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )

                if (labels == -100).all(): break

                input_ids, labels = self.unmask(input_ids, labels, out['logits'], batch_size, k, temp=temp)


            if return_logits:
                return out, input_ids, labels, out['logits']
            return out, input_ids, labels, None

        except Exception as e:
            # Clean cache on error
            self.clear_cache()
            raise e
         
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        def linear_lr_schedule(epoch):
            return 1 - epoch / self.trainer.max_epochs
        scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_schedule)
        return [optimizer], [scheduler]
        
    def training_step(self, tokens, batch_idx):
        # Assume batch contains tokens directly
        try:
            out, _, _ = self(tokens)
            # Log only the scalar value, not the tensor
            loss_value = out['loss'].item()
            self.log('loss', loss_value, sync_dist=True, prog_bar=True)
            
            # Explicitly delete tensors to free memory
            result = out['loss']
            
            # Clear memory every 100 batches
            if batch_idx % 100 == 0:
                self.clear_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return result
        except Exception as e:
            # Clean up on error
            self.clear_cache()
            raise e
            
    def on_train_epoch_end(self):
        # Clean up at the end of each epoch
        self.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    