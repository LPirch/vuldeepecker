#!/bin/bash -eu
#SBATCH --job-name=vuldeepecker
#SBATCH --partition=cpu-clx:test
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-logs/vuldeepecker/%j.out

/home/$USER/vuldeepecker/run.sh $SLURM_ARRAY_TASK_ID
