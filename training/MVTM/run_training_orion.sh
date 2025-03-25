#!/bin/bash
#SBATCH --partition a100
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 10
#SBATCH --mem 100G
#SBATCH --gres gpu:a100:8
#SBATCH --time 336:00:00
#SBATCH --job-name train-orion-mvtm-256

#Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

WANDB_DISABLE_SERVICE=True python /home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/run_training-mvtm.py \
 --config-path /home/groups/ChangLab/govindsa/checkpoints_IF/custom_vqgan_if_256.yaml\
 --ckpt-path /home/groups/ChangLab/govindsa/checkpoints_IF/epoch=000004-256.ckpt \
 --train-file /arc/scratch1/ChangLab/orion-crc/train-CRC05-06-out.h5 \
 --val-file /arc/scratch1/ChangLab/orion-crc/val-CRC05-06.h5 \
 --remove-he \
 --num-gpus 8 \
 --num-channels 17 \
 --codebook-size 256 \
 --full-channel-mask 
