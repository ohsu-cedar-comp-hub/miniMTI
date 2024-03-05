import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat,rearrange
from einops.layers.torch import Rearrange
from vit_pytorch.vit import PreNorm, FeedForward



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn_ = self.attend(dots)
        attn = self.dropout(attn_)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn_

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        attn_maps = torch.tensor([], device=x.device)
        for i, (attn, ff) in enumerate(self.layers):
            attn_x, attn_map = attn(x)
            x = attn_x + x
            x = ff(x) + x
            attn_maps = torch.concat([attn_maps, attn_map.unsqueeze(0)])
        return x, attn_maps
    

class MAE(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        channels,
        masking_ratio = 0.75,
        decoder_dim=2048,
        decoder_depth = 12,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        num_patches = channels
        patch_dim = image_size**2
        self.patch_to_emb = nn.Linear(patch_dim, decoder_dim)
        image_height, image_width = (lambda t:t  if isinstance(t, tuple) else (t, t))(image_size)
        pixel_values_per_patch = image_height * image_width
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, decoder_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, decoder_dim))

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4, dropout=0.5)
        #self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.to_pixels = nn.Sequential(nn.Linear(decoder_dim, pixel_values_per_patch), nn.ReLU())
           
        """
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, 1536),
            nn.ReLU(),
            nn.Linear(1536, pixel_values_per_patch),
            nn.ReLU()
        )"""

    def forward(self, batch, masked_patch_idx=None):
        img,mask,_  = batch
        device = img.device
        # get patches
        #patches = self.to_patch(img)
        patches = img.flatten(2,3)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens
        tokens = self.patch_to_emb(patches)
        #add positions
        tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)]
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches - 1, device = device).argsort(dim = -1) + 1 #never mask DAPI
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        #insert manually selected patches to mask
        if masked_patch_idx is not None:
            assert len(masked_patch_idx) == num_masked, f'wrong number of masked patches chosen, expected {num_masked}'
            unmasked_patch_idx = [idx for idx in range(num_patches) if idx not in masked_patch_idx]
            masked_indices = repeat(torch.tensor(masked_patch_idx), 'd -> b d', b=batch)
            unmasked_indices = repeat(torch.tensor(unmasked_patch_idx), 'd -> b d', b=batch)
            
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range].squeeze()
        
        masked_patches = patches[batch_range, masked_indices]
        
        ########### TODO: get the panel marker as well ##########
        panel_patches = patches[batch_range, unmasked_indices]

        #add cls tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch)
        cls_tokens = cls_tokens + self.pos_embedding[:, 0] #add position embedding to cls tokens
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.pos_embedding[:, masked_indices+1]

        tokens[batch_range, masked_indices+1] = mask_tokens
        decoded_tokens, attn_maps = self.decoder(tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices+1]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return panel_patches, masked_patches, pred_pixel_values, decoded_tokens, attn_maps, mask
