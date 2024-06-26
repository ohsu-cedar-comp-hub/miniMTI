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
    BATCH_SIZE = 2048
    NUM_VAL_SAMPLES = 10_000
    #train_file = f'/mnt/scratch/biolib-immune-norm/train-{val_id}-out.h5'
    train_file = f'/mnt/scratch/aced-immune-norm/train-batch-2-out.h5'
    #val_file = f'/mnt/scratch/biolib-immune-norm/biolib_immune_dataset_normed_sid={val_id}.h5'
    val_file = f'/mnt/scratch/aced-immune-norm/aced_immune_dataset_norm_sid=15-1-A-2_scene004.h5'
    train_loader, val_loader = get_train_dataloaders(train_file, val_file, BATCH_SIZE, NUM_VAL_SAMPLES)                         
    
    lr=1e-5
    image_size = 32
    num_channels = 22
    masking_ratio = 'random'
    weight_decay=0.01
    decoder_dim=1024
    decoder_depth = 6
    decoder_heads = 4
    decoder_dim_head = 64
           
    wandb_logger = WandbLogger(project="cedar-panel-reduction", entity='changlab', resume='allow', log_model=False)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    
    def get_ckpt(ckpt_id):
        dir_ = f"cedar-panel-reduction/{ckpt_id}/checkpoints/"
        fname = os.listdir(dir_)[0]
        return f"{dir_}/{fname}"
    
    ckpt = get_ckpt('7u2ixfa8')
    params = dict(lr = lr,
                   image_size=image_size,
                   num_channels=num_channels,
                   masking_ratio=masking_ratio,
                   weight_decay=weight_decay,
                   decoder_dim=decoder_dim,
                   decoder_depth = decoder_depth,
                   decoder_heads = decoder_heads,
                   decoder_dim_head = decoder_dim_head)
    
    model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    #model = IF_MAE(**params)
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

    
    
    
    
