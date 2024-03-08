import torch
import torch.nn.functional as F
from einops import repeat, rearrange
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.optim.lr_scheduler import ExponentialLR
from mae import MAE

class IF_MAE(pl.LightningModule):
    def __init__(self,
                 lr,
                 masking_ratio,
                 image_size,
                 num_channels,
                 weight_decay,
                 decoder_dim=512,
                 decoder_depth = 12,
                 decoder_heads = 8,
                 decoder_dim_head = 64):
        
        super().__init__() 
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.mae= MAE(image_size=image_size, 
                      channels=num_channels, 
                      masking_ratio = masking_ratio,
                      decoder_dim=decoder_dim,
                      decoder_depth = decoder_depth,
                      decoder_heads = decoder_heads,
                      decoder_dim_head = decoder_dim_head)

        
    def forward(self, x, masked_patch_idx):
        panel_patches, masked_patches, pred_pixel_values, encoded_tokens, attn_maps,masks = self.mae(x, masked_patch_idx=masked_patch_idx)
        return pred_pixel_values
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        _, gt, preds, _, _, mask = self.mae(train_batch)
        loss = F.mse_loss(preds, gt)
        self.log('train_loss', loss, sync_dist=True)
        
        preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
        gt = rearrange(gt, 'b c (h w) -> b c h w', h=32)
        self.logger.log_image(key="Training reconstruction", 
                       images=[preds[0][0].cpu().detach().numpy(),gt[0][0].cpu().detach().numpy()], 
                       caption=["pred", "gt"])
        
        mask = mask.bool()
        mask = repeat(mask,'b h w -> b c h w', c=preds.shape[1])
        mints = (gt * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
        pmints = (preds * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
        
        spearman_corr = spearman(mints, pmints).mean()
        ssim_score = ssim(gt, preds)
        self.log('ssim', ssim_score, sync_dist=True)
        self.log('spearman', spearman_corr, sync_dist=True)
        
        return loss
    
    def validation_step(self, val_batch,  val_idx):
        _, gt, preds, _, _, mask = self.mae(val_batch)
        loss = F.mse_loss(preds, gt)
        self.log('val_loss', loss, sync_dist=True)
        
        preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
        gt = rearrange(gt, 'b c (h w) -> b c h w', h=32)
        self.logger.log_image(key="Val reconstruction", 
                       images=[preds[0][0].cpu().detach().numpy(),gt[0][0].cpu().detach().numpy()], 
                       caption=["pred", "gt"])
        
        mask = mask.bool()
        mask = repeat(mask,'b h w -> b c h w', c=preds.shape[1])
        mints = (gt * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
        pmints = (preds * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
        
        spearman_corr = spearman(mints, pmints).mean()
        ssim_score = ssim(gt, preds)
        self.log('val_ssim', ssim_score, sync_dist=True)
        self.log('val_spearman', spearman_corr, sync_dist=True)