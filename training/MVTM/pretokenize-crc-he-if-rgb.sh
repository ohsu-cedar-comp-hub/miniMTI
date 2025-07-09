#!/bin/bash
#SBATCH --partition a100
#SBATCH --output=pretokenize_output%j.out
#SBATCH --error=pretokenize_error%j.err
#SBATCH --cpus-per-task 30
#SBATCH --mem 500G
#SBATCH --gres gpu:a100:1
#SBATCH --time 72:00:00
#SBATCH --job-name pretokenize-data

# Activate conda environment
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

# Run the pre-tokenization script
python /home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/pretokenize_data.py \
  --config-path-if /home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/configs/custom_vqgan_if_only_256.yaml \
  --ckpt-path-if /home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/vqgan_checkpoints_opt/if_only_he_rgb_vqgan_models/if_only_256_ckpts/last.ckpt \
  --config-path-he /home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/configs/custom_vqgan_he_rgb_256.yaml \
  --ckpt-path-he /home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/vqgan_checkpoints_opt/if_only_he_rgb_vqgan_models/he_rgb_256_ckpts/last.ckpt \
  --train-file /home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All/train-all-out.h5 \
  --val-file /home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All/val-all.h5 \
  --output-dir /home/groups/CEDAR/panel-reduction-project/tokenized-crc-data \
  --num-channels 20 \
  --codebook-size 256 \
  --batch-size 64