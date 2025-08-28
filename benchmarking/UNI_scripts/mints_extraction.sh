#!/bin/bash
#SBATCH --partition gpu
#SBATCH --output=error_logs_mints/train_output_vae%j.out        ### File in which to store job output
#SBATCH --error=error_logs_mints/train_fine_error_vae%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:a40:8
#SBATCH --time=336:00:00
#SBATCH --job-name train-fine_mints

# Activate conda environment:
# echo "Initializing env"
source /home/groups/ChangLab/govindsa/miniconda3/bin/activate visenv
# echo "Initialized env"
# WANDB_DISABLE_SERVICE=True python run.py --train_config config/train_config.py --model_config config/beta_vae.py
# # python embed.py --train_config config/train_config.py --model_config config/beta_vae.py
python mean_intensity_extraction.py