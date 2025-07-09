#!/bin/bash
#SBATCH --partition gpu
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 30
#SBATCH --mem 100G
#SBATCH --gres gpu:a40:4
#SBATCH --time 168:00:00
#SBATCH --job-name train-tokenized-mvtm

# Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

# Run the training script with pre-tokenized data
WANDB_DISABLE_SERVICE=True WANDB_RUN_ID=hhdnp2ma python run_training_tokenized.py --train-file /home/exacloud/gscratch/ChangLab/tokenized-crc-data-individual/train/train.h5 --num-gpus 4 --batch-size 64 --num-workers 1 --num-epochs 20 --use-checkpoint --precision 16 --gradient-accumulation 1
