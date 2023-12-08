#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=mocs_exp_%j
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=1:00:00

set -x

echo "$1" "$2"
python3 run_exp_vacc.py "$1" "$2"