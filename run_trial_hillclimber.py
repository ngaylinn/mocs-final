from pathlib import Path
import tracemalloc
import linecache
import os
import time
import argparse

import numpy as np

from hillclimber import HillClimber
from simulation import visualize

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_file', type=str, help='Name of experiment file')
parser.add_argument('exp_name', type=str, help='Name of experiment')
parser.add_argument('arm_name', type=str, help='Name of experiment arm')
parser.add_argument('trial', type=int, help='Trial number')
args = parser.parse_args()

# Read the experiment file into exp_arms variable
exp_file = open(args.exp_file)
exp_string = exp_file.read()
exp_arms = eval(exp_string)
exp_file.close()

print(exp_arms)
# Get the parameters for this particular arm
arm_parameters = exp_arms[args.arm_name]

# Create experiment directory if it doesn't already exist
if not os.path.exists(f'./experiments/{args.exp_name}'):
    os.system(f'mkdir ./experiments/{args.exp_name}')

# Copy experiment file into experiment directory
os.system(f'cp {args.exp_file} ./experiments/{args.exp_name}')

def main():
    # Create an arm directory if it doesn't already exist
    if not os.path.exists(f'./experiments/{args.exp_name}/{args.arm_name}'):
        os.system(f'mkdir ./experiments/{args.exp_name}/{args.arm_name}')

    single_run = HillClimber(arm_parameters)
    single_run.evolve()

    single_run.pickle_hc(f'./experiments/{args.exp_name}/{args.arm_name}/{args.arm_name}_t{args.trial}.pkl')

if __name__ == '__main__':
    main()
