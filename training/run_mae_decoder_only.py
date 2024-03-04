import os
import sys
import pickle
import torch
import wandb
import numpy as np
from skimage.io import imread
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from mae_decoder_only_train import MAE
from einops import repeat, rearrange
from torch.utils.data import Dataset
import random
from data import get_train_dataloaders
    
    
class IF_MAE(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay,
                 warmup_steps,
                 decoder_dim=512,
                 decoder_depth = 12,
                 decoder_heads = 8,
                 decoder_dim_head = 64):
        
        super().__init__() 
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_warmup_steps = warmup_steps
        
        self.mae= MAE(image_size=32, 
                      channels=41, 
                      masking_ratio = 0.5,
                      decoder_dim=decoder_dim,
                      decoder_depth = decoder_depth,
                      decoder_heads = decoder_heads,
                      decoder_dim_head = decoder_dim_head)

        
    def forward(self, x, masked_patch_idx):
        masked_patches, pred_pixel_values = self.mae(x, masked_patch_idx=masked_patch_idx)
        return pred_pixel_values
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #scheduler = ExponentialLR(optimizer, self.lr_warmup_steps)
        return optimizer
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'train_loss',
                }
            ]
        )


    def training_step(self, train_batch, batch_idx):
        gt, preds, mask = self.mae(train_batch)
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
        gt, preds, mask = self.mae(val_batch)
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
    
    
def train_model():
    BATCH_SIZE = 2048
    remove_background = True
    NUM_VAL_SAMPLES = 50_000
    include_he = False
    grayscale = False
    deconvolve = False
    train_loader, val_loader, val2_loader = get_train_dataloaders(BATCH_SIZE, NUM_VAL_SAMPLES, include_he, remove_background, grayscale, deconvolve)
                            
    
    lr=1e-4
    weight_decay=0
    warmup_steps=50
    decoder_dim=1024
    decoder_depth = 11
    decoder_heads = 6
    decoder_dim_head = 64
           
    wandb_logger = WandbLogger(project="cedar-panel-reduction", entity='changlab', resume='allow', log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    
    model = IF_MAE(decoder_dim=decoder_dim,
                   decoder_depth = decoder_depth,
                   decoder_heads = decoder_heads,
                   decoder_dim_head = decoder_dim_head,
                   lr = lr,
                   weight_decay=weight_decay,
                   warmup_steps=warmup_steps)
    wandb_logger.watch(model, log="all")
    
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(69, workers=True)
    
    trainer = pl.Trainer(accelerator='gpu',
                         devices=4,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=150,
                         num_sanity_val_steps=0,
                         strategy='ddp')
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update({"learning_rate":lr,
                                               "weight_decay":weight_decay,
                                               "decoder_dim":decoder_dim,
                                               "decoder_depth":decoder_depth,
                                               "decoder_heads":decoder_heads,
                                               "decoder_dim_head":decoder_dim_head,
                                               })
    trainer.fit(model, train_loader, val_loader)
    
        
if __name__ == '__main__':
    #sweep_id = sys.argv[1]
    #wandb.agent(sweep_id=sweep_id, function=train_model, count=20, project='pt_mae', entity='changlab')
    
    train_model()

    
    
    
    
