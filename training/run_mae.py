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
    NUM_EPOCHS = 150
    BATCH_SIZE = 1024
    NUM_VAL_SAMPLES = 50_000
    train_file = '/var/local/ChangLab/biolib-immune/train.h5'
    val_file = '/var/local/ChangLab/biolib-immune/biolib_immune_dataset_rescaled_sid=57658-6.h5'
    train_loader, val_loader = get_train_dataloaders(train_file, val_file, BATCH_SIZE, NUM_VAL_SAMPLES)                         
    
    lr=1e-4
    image_size = 32
    num_channels = 40
    masking_ratio = 0.5
    weight_decay=0
    decoder_dim=1024
    decoder_depth = 8
    decoder_heads = 4
    decoder_dim_head = 64
           
    wandb_logger = WandbLogger(project="cedar-panel-reduction", entity='changlab', resume='allow', log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    
    model = IF_MAE(lr = lr,
                   image_size=image_size,
                   num_channels=num_channels,
                   masking_ratio=masking_ratio,
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
    train_model()

    
    
    
    
