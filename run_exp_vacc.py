from pathlib import Path
import tracemalloc
import linecache
import os
import time
import argparse

import numpy as np

from afpo import AgeFitnessPareto
from simulation import visualize

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_file', type=str, help='Experiment file')
parser.add_argument('name', type=str, help='Name of experiment')
parser.add_argument('optimizer', default='afpo', type=str, help='Name of optimizer')
args = parser.parse_args()

# Read the experiment file into exp_arms variable
exp_file = open(args.exp_file)
exp_string = exp_file.read()
exp_arms = eval(exp_string)
exp_file.close()

# Create experiment directory if it doesn't exist
if not os.path.exists('./experiments'):
    os.system('mkdir experiments')
if not os.path.exists(f'./experiments/{args.name}'):
    os.system(f'mkdir ./experiments/{args.name}')

# Copy experiment file into experiment directory
os.system(f'cp {args.exp_file} ./experiments/{args.name}')

def main():
    for arm in exp_arms:
        # Create an arm directory if it doesn't already exist
        if not os.path.exists(f'./experiments/{args.name}/{arm}'):
            os.system(f'mkdir ./experiments/{args.name}/{arm}')

        arm_parameters = exp_arms[arm]

        # Run a few instances of each
        n_trials = arm_parameters['num_trials']
        for trial in range(n_trials):
            if args.optimizer == 'afpo':
                os.system(f'sbatch vacc_submit_arm.sh {args.exp_file} {args.name} {arm} {trial}')
            elif args.optimizer == 'hillclimber':
                os.system(f'sbatch vacc_submit_arm_hc.sh {args.exp_file} {args.name} {arm} {trial}')


if __name__ == '__main__':
    main()
