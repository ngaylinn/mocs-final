from pathlib import Path
import tracemalloc
import linecache
import os
import time
import argparse

import numpy as np

from afpo import Solution
from hillclimber import HillClimber
from simulation import simulate, make_seed_phenotypes, visualize, generate_random_above_map, generate_random_below_map, do_they_percolate

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

def main(above_maps, below_maps):
    exp_directory = f'./experiments/{args.exp_name}/{args.arm_name}'
    # Create an arm directory if it doesn't already exist
    if not os.path.exists(exp_directory):
        os.system(f'mkdir {exp_directory}')

    single_run = HillClimber(arm_parameters, above_maps, below_maps, exp_directory)

    single_run.evolve()

    single_run.pickle_hc(f'./experiments/{args.exp_name}/{args.arm_name}/{args.arm_name}_t{args.trial}.pkl')

def get_above_below_maps(n=500):
    n_layers = 4
    above_maps = np.zeros((500, 4, 3))
    below_maps = np.zeros((500, 4, 4, 3))

    n_generated = 0
    
    # Generate random above/below maps
    prospective_above_maps = np.array([generate_random_above_map() for _ in range(n)])
    prospective_below_maps = np.array([generate_random_below_map() for _ in range(n)])

    sol = Solution(layers=[{'res': 1}, {'res': 2}, {'res': 4}, {'res': 8, 'base': True}], id=0, above_map=prospective_above_maps[0], below_map=prospective_below_maps[0])

    while n_generated < 500:
        # Get initial phenotypes
        seed_phenotypes = make_seed_phenotypes(500)

        # All ones genotypes to allow percolation...
        genotypes = np.ones((n, 4, 14))
         
        # Simulate them w/ all ones genotypes... 
        phenotypes = simulate(
                genotypes, 
                n_layers, 
                sol.around_start, 
                sol.above_start, 
                seed_phenotypes, 
                prospective_below_maps,
                prospective_above_maps)
        
        percolators = do_they_percolate(phenotypes)
        n_percolators = sum(percolators)

        percolator_above_maps = prospective_above_maps[percolators]
        percolator_below_maps = prospective_below_maps[percolators]
        
        print(len(percolator_above_maps))

        for i in range(n_percolators):
            above_maps[n_generated] = percolator_above_maps[i]
            below_maps[n_generated] = percolator_below_maps[i]
            n_generated += 1
            if n_generated == 500:
                break
        
    return above_maps, below_maps
        

if __name__ == '__main__':
    print('Generating viable above and below maps')
    above_maps, below_maps = get_above_below_maps()
    # print(below_maps[0][0][0][0])
    print('Above and below maps generated. Commencing evolution...')
    main(above_maps.astype(np.int32), below_maps.astype(np.int32))
 