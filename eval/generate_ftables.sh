#!/bin/bash
#SBATCH --partition gpu
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 10
#SBATCH --mem 100G
#SBATCH --gres gpu:a40:1
#SBATCH --time 168:00:00
#SBATCH --job-name gen-ftables

#Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate vqgan

python generate_ftable_from_h5_tokenized.py 01 1
