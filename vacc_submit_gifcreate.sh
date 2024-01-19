#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=gif_create_%j
#SBATCH --output=./vacc_out_files/gif_%x_%j.out
#SBATCH --time=00:02:00

set -x

python3 vis.py --exp experiments/exp_diamond_3layer/base1_diamond_64.32.16/base1_diamond_64.32.16_t0.pkl

python3 vis.py --exp experiments/exp_diamond_3layer/base1_diamond_64.32.16/base1_diamond_64.32.16_t1.pkl

python3 vis.py --exp experiments/exp_diamond_3layer/base1_diamond_64.32.16/base1_diamond_64.32.16_t2.pkl

python3 vis.py --exp experiments/exp_diamond_3layer/base1_diamond_64.32.16/base1_diamond_64.32.16_t3.pkl

python3 vis.py --exp experiments/exp_diamond_3layer/base1_diamond_64.32.16/base1_diamond_64.32.16_t4.pkl