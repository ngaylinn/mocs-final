#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=6:00:00
#SBATCH --job-name=mocs_exp_%x

set -x

python3 run_exp.py --exp_file experiment_example.txt --name ex_experiment