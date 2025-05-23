#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=hnca_%j
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=9:00:00

set -x

echo $1
echo $2
echo $3
echo $4

conda activate hnca-cuda

python3 -O run_trial.py "$1" "$2" "$3" "$4"