import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tracemalloc
import argparse

from simulation import simulate
from neutral_engine import NeutralEngine
from vis import visualize_all_layers, visualize_layers_and_selection_over_time

def get_experiment(exp_file):
    with open(exp_file, 'rb') as pf:
        exp = pickle.load(pf)

    return exp

def simulate_one_individual(exp, solution):
    init_phenotypes = exp.make_seed_phenotypes(1)
    phenotypes = simulate(
            np.array([solution.state_genotype]),
            solution.n_layers,  
            solution.around_start, 
            solution.above_start, 
            phenotypes=init_phenotypes,
            below_map=np.array(exp.below_map),
            above_map=np.array(exp.above_map))

    # phenotypes = simulate( # NONLOCAL UPDOWN
    #         np.array([solution.state_genotype]),
    #         solution.n_layers,  
    #         solution.around_start, 
    #         solution.above_start, 
    #         init_phenotypes, 
    #         np.array([solution.below_map]),
    #         np.array([solution.above_map]))
    
    return phenotypes

parser = argparse.ArgumentParser()
parser.add_argument('exp_file', type=str, help='Experiment File')
args = parser.parse_args()

# ne_old = get_experiment('./neutral_walker_Apr15_circle_betta2.pkl')
# best = ne_old.beneficial_solutions[0]

exp = get_experiment(args.exp_file)
best = exp.best_solution()

best.neutral_counter = 0
print('best fitness: ', best.fitness)

ne = NeutralEngine(exp, best, best.state_genotype)

# Code
ne.run(5)

ne.pickle_ne('neutral_walker_Sep2.pkl')

##### 

signaling_dist, sig_sol = ne.longest_signaling_distance_from_original()
neutral_walk_length, neutral_sol = ne.longest_neutral_walk_from_original()


print('Longest signaling distance: ', signaling_dist, f' (neutral path length: {sig_sol.neutral_counter})')
print('Longest neutral path: ', neutral_walk_length, f' (signaling distance: {neutral_sol.signaling_distance})')

phenotypes_max_signaling = simulate_one_individual(exp, sig_sol)
phenotypes_orig = simulate_one_individual(exp, ne.init_solution)
phenotypes_max_neutral_walk = simulate_one_individual(exp, neutral_sol)



visualize_all_layers(phenotypes_orig[0], './neutral_walk/orig.gif', base_layer_idx=exp.base_layer)
visualize_all_layers(phenotypes_max_signaling[0], './neutral_walk/max_signaling.gif', base_layer_idx=exp.base_layer)
visualize_all_layers(phenotypes_max_neutral_walk[0], './neutral_walk/max_neutralwalk.gif', base_layer_idx=exp.base_layer)

'''
# frames = [5, 10, 25, 99]
# frames = [5, 10, 25, 50, 75, 95, 99]
frames = [5, 10, 20, 30, 40, 50, 60, 70, 80, 95, 99]
visualize_layers_and_selection_over_time(phenotypes_orig[0], './neutral_walk/original.png', 3, 'complex', color='blue', frames = frames)
visualize_layers_and_selection_over_time(phenotypes_max_neutral_walk[0], './neutral_walk/max_neutral_walk.png', 3, 'complex', color='blue', frames = frames)
visualize_layers_and_selection_over_time(phenotypes_max_signaling[0], './neutral_walk/max_signaling.png', 3, 'complex', color='blue', frames = frames)

if len(ne.beneficial_solutions) > 0:
    beneficial_sol = ne.beneficial_solutions[0]
    phenotypes_beneficial = simulate_one_individual(exp, beneficial_sol)
    visualize_all_layers(phenotypes_beneficial[0], './neutral_walk/beneficial_neutralwalk.gif', base_layer_idx=exp.base_layer)
    visualize_layers_and_selection_over_time(phenotypes_beneficial[0], './neutral_walk/beneficial.png', 3, 'complex', color='blue', frames = frames)
'''
