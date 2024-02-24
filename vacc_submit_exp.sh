#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=mocs_exp_%j
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=1:00:00

set -x

# Arg 1: Experiment file
# Arg 2: Experiment name
# Arg 3: optimizer type ('afpo' or 'hillclimber')

echo "$1" "$2" "$3"
python3 run_exp_vacc.py "$1" "$2" "$3"