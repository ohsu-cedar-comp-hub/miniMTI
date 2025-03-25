import os
import sys
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from train_mae import IF_MAE
sys.path.append('../data')
from data import get_train_dataloaders
    
    
def train_model(val_id):
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    NUM_VAL_SAMPLES = 10_000
    #train_file = '/mnt/scratch/ORION-CRC-Unnormalized/train-CRC05-06-out.h5'
    #val_file = '/mnt/scratch/ORION-CRC-Unnormalized/val-CRC05-06.h5'
    #train_file = '/arc/scratch1/ChangLab/orion-crc/train-CRC05-06-out.h5'
    #val_file = '/arc/scratch1/ChangLab/orion-crc/val-CRC05-06.h5'
    train_file = '/arc/scratch1/ChangLab/aced-immune-norm-40mx/train-batch6-out.h5'
    val_file = '/arc/scratch1/ChangLab/aced-immune-norm-40mx/val-batch6.h5'
    train_loader, val_loader = get_train_dataloaders(train_file, val_file, BATCH_SIZE, NUM_VAL_SAMPLES, remove_he=False, downscale=True, deconvolve_he=False, remove_background=False, rescale=False)                         
    
    lr=1e-4
    image_size = 32
    num_channels = 41
    masking_ratio = 'random'
    weight_decay=0.001
    decoder_dim=2048
    decoder_depth = 8
    decoder_heads = 6
    decoder_dim_head = 64
           
    wandb_logger = WandbLogger(project="cedar-panel-reduction", entity='changlab', resume='allow', log_model=False)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    
    def get_ckpt(ckpt_id):
        dir_ = f"cedar-panel-reduction/{ckpt_id}/checkpoints/"
        fname = os.listdir(dir_)[0]
        return f"{dir_}/{fname}"
    
    ckpt = get_ckpt('6vtoavjz')
    params = dict(lr = lr,
                   image_size=image_size,
                   num_channels=num_channels,
                   masking_ratio=masking_ratio,
                   weight_decay=weight_decay,
                   decoder_dim=decoder_dim,
                   decoder_depth = decoder_depth,
                   decoder_heads = decoder_heads,
                   decoder_dim_head = decoder_dim_head)
    
    #model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    model = IF_MAE(**params)
    wandb_logger.watch(model, log="all")
    
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(69, workers=True)
    
    trainer = pl.Trainer(accelerator='gpu',
                         devices=4,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=NUM_EPOCHS,
                         num_sanity_val_steps=0,
                         strategy='ddp')
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update({"learning_rate":lr,
                                               "weight_decay":weight_decay,
                                               "decoder_dim":decoder_dim,
                                               "decoder_depth":decoder_depth,
                                               "decoder_heads":decoder_heads,
                                               "decoder_dim_head":decoder_dim_head})
    trainer.fit(model, train_loader, val_loader)
    
    
if __name__ == '__main__':
    #val_id = sys.argv[1]
    train_model(val_id='15-1-A-2_scene004')

    
    
    
    
