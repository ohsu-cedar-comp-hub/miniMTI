import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from mvtm_updated import IF_MVTM
sys.path.append('/home/groups/ChangLab/govindsa/cycif-panel-reduction/data/')
from data import get_train_dataloaders
pl.seed_everything(69, workers=True)

# Argument parsing for command line input
parser = argparse.ArgumentParser(description='Run training with MVTM')
parser.add_argument('--config_path', type=str, required=True, help='Path to VQGAN config file')
parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the VQGAN checkpoint file')
parser.add_argument('--vq_dim', type=int, default=4, help='Dimension of VQ latent space (default: 4)')
parser.add_argument('--vq_f_dim', type=int, default=256, help='Feature dimension for VQGAN (default: 256)')
parser.add_argument('--train_file', type=str, default='/mnt/scratch/ORION-CRC-Unnormalized/train-CRC05-06-out.h5', help='Path to the training dataset file')
parser.add_argument('--val_file', type=str, default='/mnt/scratch/ORION-CRC-Unnormalized/orion_crc_dataset_sid=CRC05.h5', help='Path to the validation dataset file')
args = parser.parse_args()

def get_ckpt(ckpt_id):
    print('entered get_ckpt stage')
    dir_ = f"VQ-panel-reduction/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    print('finished get_ckpt stage')
    return f"{dir_}/{fname}"

def train_model(config_path, ckpt_path, vq_dim, vq_f_dim):
    print('entered train model')
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    NUM_VAL_SAMPLES = 10_000
    # train_file = '/mnt/scratch/ORION-CRC-Unnormalized/train-CRC05-06-out.h5'
    # val_file = '/mnt/scratch/ORION-CRC-Unnormalized/orion_crc_dataset_sid=CRC05.h5'
    train_file=args.train_file
    val_file=args.val_file
    print('Before train_loader')
    train_loader, val_loader = get_train_dataloaders(train_file, val_file, BATCH_SIZE, NUM_VAL_SAMPLES)
    print('after train loader')

    params = dict(
        lr=4e-4,
        weight_decay=0.001,
        num_channels=17,
        num_layers=24,
        num_heads=8,
        latent_dim=768,
        num_codes=1024,
        config_path=config_path,
        ckpt_path=ckpt_path,
        vq_dim=vq_dim,
        vq_f_dim=vq_f_dim
    )
    print('getting model details')
    model = IF_MVTM(**params)
    print('starting wandb')
    wandb_logger = WandbLogger(project="MVTM-panel-reduction", entity='changlab', resume='allow', log_model=False)
    wandb_logger.watch(model, log="all")
    print('start checkpoint callback')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    print('checkpoint callback done')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=NUM_EPOCHS,
        num_sanity_val_steps=1,
        strategy='ddp'
    )
    print('trainer done')
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(params)
    print('starting fitting model')
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train_model(args.config_path, args.ckpt_path, args.vq_dim, args.vq_f_dim)
