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

    
    
    
    
