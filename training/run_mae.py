import sys
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from train_mae import IF_MAE
sys.path.append('../data')
from data import get_train_dataloaders
    
    
def train_model():
    BATCH_SIZE = 1024
    NUM_VAL_SAMPLES = 50_000
    data_dir = '/var/local/ChangLab/biolib-immune'
    val_samples = ['57658-6']
    train_loader, val_loader = get_train_dataloaders(data_dir, val_samples, BATCH_SIZE, NUM_VAL_SAMPLES)                         
    
    lr=1e-4
    weight_decay=0
    decoder_dim=1024
    decoder_depth = 8
    decoder_heads = 4
    decoder_dim_head = 64
           
    wandb_logger = WandbLogger(project="cedar-panel-reduction", entity='changlab', resume='allow', log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    
    model = IF_MAE(lr = lr,
                   weight_decay=weight_decay,
                   decoder_dim=decoder_dim,
                   decoder_depth = decoder_depth,
                   decoder_heads = decoder_heads,
                   decoder_dim_head = decoder_dim_head)
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
                                               "decoder_dim_head":decoder_dim_head})
    trainer.fit(model, train_loader, val_loader)
    
    
if __name__ == '__main__':
    train_model()

    
    
    
    
