#!/bin/bash
#SBATCH --partition batch
#SBATCH --output=output%j.out         ### File in which to store job output
#SBATCH --error=error%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 1
#SBATCH --mem 500G
#SBATCH --time 18:00:00
#SBATCH --job-name preprocess-data

# Activate conda environment:
source /home/groups/ChangLab/simsz/conda/bin/activate wsi_process

/usr/bin/time -v python /home/groups/ChangLab/simsz/cycif-panel-reduction/data/process_lunaphore.py -j $SLURM_JOB_ID



