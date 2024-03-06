import sys
import pytorch_lightning as pl
sys.path.append('../training')
from mae import MAE

class IF_MAE(pl.LightningModule):
    def __init__(self, 
                 channels,
                 decoder_dim=512,
                 decoder_depth = 4,
                 decoder_heads = 2,
                 decoder_dim_head = 64):
        super().__init__()
        self.save_hyperparameters()
        self.mae= MAE(image_size=32, 
                      channels=channels, 
                      decoder_dim=decoder_dim,
                      decoder_depth=decoder_depth,
                      decoder_heads=decoder_heads,
                      decoder_dim_head=decoder_dim_head)
        
    def forward(self, x, masked_patch_idx):
        panel_patches, masked_patches, pred_pixel_values, encoded_tokens, attn_maps,masks = self.mae(x, masked_patch_idx=masked_patch_idx)
        return panel_patches, masked_patches, pred_pixel_values, encoded_tokens, attn_maps