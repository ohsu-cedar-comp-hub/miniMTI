#!/bin/bash
#SBATCH --partition a100
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 10
#SBATCH --mem 100G
#SBATCH --gres gpu:a100:8
#SBATCH --time 168:00:00
#SBATCH --job-name train-aced-mvtm-a100

#Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

python /home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/run_training-mvtm.py \
 --config-path /home/groups/ChangLab/simsz/latent-diffusion/src/taming-transformers/configs/custom_vqgan2.yaml \
 --ckpt-path /home/groups/ChangLab/simsz/latent-diffusion/src/taming-transformers/vqgan-aced-ckpts/last-v1.ckpt \
 --train-file /arc/scratch1/ChangLab/aced-immune-norm-40mx/train-batch6-out.h5 \
 --val-file /arc/scratch1/ChangLab/aced-immune-norm-40mx/val-batch6.h5 \
 --num-channels 40 \
 --downscale \
 --codebook-size 256 \
 --num-gpus 8
