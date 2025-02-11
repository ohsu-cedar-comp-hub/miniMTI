#!/bin/bash
#SBATCH --partition a100
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 10
#SBATCH --mem 50G
#SBATCH --gres gpu:a100:8
#SBATCH --time 168:00:00
#SBATCH --job-name train-orion-mvtm-he-rgb

#Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

#run script to create train set and test set h5 files
python ../data/create_training_files.py \
--data-dir /arc/scratch1/ChangLab/lunaphore-immune-unnorm/ \
--val-samples /home/groups/ChangLab/simsz/cycif-panel-reduction/data/lunaphore_val_samples.txt \
--batch-name LUN-58-76

#run training script
WANDB_DISABLE_SERVICE=True python ../training/MVTM/run_training-mvtm.py \
 --config-path-if /home/groups/ChangLab/simsz/latent-diffusion/src/taming-transformers/configs/custom_vqgan.yaml\
 --ckpt-path-if /home/groups/ChangLab/simsz/latent-diffusion/src/taming-transformers/vqgan-aced-ckpts/last-v21.ckpt \
 --train-file /arc/scratch1/ChangLab/lunaphore-immune-unnorm/train-LUN-58-76-out.h5 \
 --val-file /arc/scratch1/ChangLab/lunaphore-immune-unnorm/val-LUN-58-76.h5 \
 --num-gpus 8 \
 --num-channels 43 \
 --codebook-size 256 \
 --downscale \
 --full-channel-mask 