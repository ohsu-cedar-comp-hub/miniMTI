#!/bin/bash
#SBATCH --partition exacloud
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 1
#SBATCH --mem 400G
#SBATCH --time 12:00:00
#SBATCH --job-name preprocess-biolib

# Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate wsi_process

/usr/bin/time -v python /home/groups/ChangLab/simsz/cycif-panel-reduction/data/process_orion_crc.py -j $SLURM_JOB_ID



