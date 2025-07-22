#!/bin/bash
#SBATCH --partition gpu
#SBATCH --output=output_rev.out         ### File in which to store job output
#SBATCH --error=error_rev.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 2
#SBATCH --mem 200G
#SBATCH --gres gpu:a40:1
#SBATCH --time 30:00:00
#SBATCH --job-name rev-panel-select

# Activate conda environment
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

#python create_panel_selection_data.py

python run_panel_selection_tokenized.py \
--val-dataset /home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All/orion_panel_select_data_CRC02_CRC03.h5 \
--param-file /home/groups/ChangLab/simsz/cycif-panel-reduction/eval/configs/params_mvtm-256-he-rgb-tokenized.json \
--max-panel-size 18 \
--gpu-id 0 \
--reversed