#!/bin/bash -eu
#SBATCH --job-name=vuldeepecker
#SBATCH --partition=cpu-2h
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-logs/vuldeepecker/%j.out

/home/$USER/vuldeepecker/scripts/run.sh
