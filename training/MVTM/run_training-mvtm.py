import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from mvtm import IF_MVTM
sys.path.append('../../data')
from data import get_train_dataloaders
pl.seed_everything(69, workers=True)
 
def get_ckpt(ckpt_id):
    dir_ = f"VQ-panel-reduction/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    return f"{dir_}/{fname}"
    
def train_model():
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    NUM_VAL_SAMPLES = 10_000
    train_file = '/mnt/scratch/aced-immune-norm-40mx/train-batch6-out.h5'
    val_file = '/mnt/scratch/aced-immune-norm-40mx/val-batch6.h5'
    train_loader, val_loader = get_train_dataloaders(train_file, val_file, BATCH_SIZE, NUM_VAL_SAMPLES)   
    
    params = dict(
        lr=4e-4,
        weight_decay=0.001,
        num_channels = 40,
        num_layers = 12,
        num_heads = 12,
        latent_dim = 768,
        num_codes=256
    )
    
    model = IF_MVTM(**params)
    
    wandb_logger = WandbLogger(project="MVTM-panel-reduction", entity='changlab', resume='allow', log_model=False)
    wandb_logger.watch(model, log="all")
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=NUM_EPOCHS,
        num_sanity_val_steps=1,
        strategy='ddp'
    )
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(params)
        
    trainer.fit(model, train_loader, val_loader)
    
    
if __name__ == '__main__':
    train_model()

    
    
    
    
