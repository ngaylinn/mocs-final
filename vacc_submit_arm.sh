#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=mocs_exp_%j
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=6:00:00

set -x

python3 run_trial.py "$1" "$2" "$3" "$4"