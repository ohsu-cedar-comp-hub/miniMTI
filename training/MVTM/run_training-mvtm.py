import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from mvtm import IF_MVTM
sys.path.append('../../data/')
from data import get_train_dataloaders
pl.seed_everything(69, workers=True)

# Argument parsing for command line input
parser = argparse.ArgumentParser(description='Run training with MVTM')
parser.add_argument('--config-path', type=str, required=True, help='Path to VQGAN config file')
parser.add_argument('--ckpt-path', type=str, required=True, help='Path to the VQGAN checkpoint file')
parser.add_argument('--vq-dim', type=int, default=4, help='Dimension of VQ latent space (default: 4)')
parser.add_argument('--vq-f-dim', type=int, default=256, help='Feature dimension for VQGAN (default: 256)')
parser.add_argument('--train-file', type=str, default='/mnt/scratch/ORION-CRC-Unnormalized/train-CRC05-06-out.h5', help='Path to the training dataset file')
parser.add_argument('--val-file', type=str, default='/mnt/scratch/ORION-CRC-Unnormalized/orion_crc_dataset_sid=CRC05.h5', help='Path to the validation dataset file')
parser.add_argument('--remove-he', action='store_true', help="remove last three channels if H&E is stored with IF")
parser.add_argument('--downscale', action='store_true', help="downscale images 2x")
parser.add_argument('--codebook-size', type=int, default=1024, help="number of VQ codes")
parser.add_argument('--num-channels', type=int, required=True, help="number of channels per image")
parser.add_argument('--num-gpus', type=int, required=True, help="number of GPUs to use")
args = parser.parse_args()

def get_ckpt(ckpt_id):
    print('entered get_ckpt stage')
    dir_ = f"VQ-panel-reduction/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    print('finished get_ckpt stage')
    return f"{dir_}/{fname}"

def train_model(config_path, ckpt_path, vq_dim, vq_f_dim, remove_he, downscale, codebook_size, num_channels, num_gpus):
    print('entered train model')
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    NUM_VAL_SAMPLES = 10_000
    train_file=args.train_file
    val_file=args.val_file
    print('Before train_loader')
    train_loader, val_loader = get_train_dataloaders(train_file, val_file, BATCH_SIZE, NUM_VAL_SAMPLES, remove_he=remove_he, downscale=downscale)
    print('after train loader')

    params = dict(
        lr=1e-5,
        weight_decay=0,
        num_channels=num_channels,
        num_layers=24,
        num_heads=8,
        latent_dim=768,
        num_codes=codebook_size,
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
    checkpoint_callback = ModelCheckpoint(monitor="loss", mode="min")
    print('checkpoint callback done')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=NUM_EPOCHS,
        num_sanity_val_steps=1,
        strategy='ddp',
        default_root_dir="/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM"
    )
    print('trainer done')
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(params)
    print('starting fitting model')
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train_model(args.config_path, args.ckpt_path, args.vq_dim, args.vq_f_dim, args.remove_he, args.downscale, args.codebook_size, args.num_channels, args.num_gpus)
