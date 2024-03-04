import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from vit_pytorch.vit import Transformer

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
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4, dropout=0.1)
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
        img, mask,_ = batch
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
        num_masked = int(self.masking_ratio * (num_patches - 1))
        rand_indices = torch.rand(batch, num_patches - 1, device = device).argsort(dim = -1) + 1 #never mask DAPI/HE
        #rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
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

        #add cls tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch)
        cls_tokens = cls_tokens + self.pos_embedding[:, 0] #add position embedding to cls tokens
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.pos_embedding[:, masked_indices+1]

        tokens[batch_range, masked_indices+1] = mask_tokens
        decoded_tokens = self.decoder(tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices+1]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return masked_patches, pred_pixel_values, mask

if __name__ == '__main__':
    #x = torch.rand(2,1,160,160,)
    x = torch.rand(1024,17,32,32)
    mae = MAE(image_size=32,
              channels=17,
              masking_ratio = 0.5,   
              decoder_dim = 512,       
              decoder_depth = 6)   
    mae(x)[0].shape
