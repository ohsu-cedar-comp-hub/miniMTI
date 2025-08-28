#!/bin/bash
#SBATCH --partition gpu
#SBATCH --output=error_logs_regressor/crc071310_output_vae%j.out        ### File in which to store job output
#SBATCH --error=error_logs_regressor/crc071310_error_vae%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task=12
#SBATCH --mem=500G
#SBATCH --gres=gpu:a40:8
#SBATCH --time=336:00:00
#SBATCH --job-name crc071310_regressor

# Activate conda environment:
# echo "Initializing env"
source /home/groups/ChangLab/govindsa/miniconda3/bin/activate visenv
# echo "Initialized env"
# WANDB_DISABLE_SERVICE=True python run.py --train_config config/train_config.py --model_config config/beta_vae.py
# # python embed.py --train_config config/train_config.py --model_config config/beta_vae.py
python training_regressor_uni.py