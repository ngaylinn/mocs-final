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
parser.add_argument('--exp_file', type=str, help='Experiment File')
parser.add_argument('--walk_type', type=str, help='Walk Type', default='random', choices=['random', 'genotype_distance'])
args = parser.parse_args()

exp = get_experiment(args.exp_file)
best = exp.best_solution()

print('best fitness: ', best.fitness)

best.genotype_distance = 0
ne = NeutralEngine(exp, best, best.state_genotype, mutate_layers=None, walk_type=args.walk_type)
ne.run(30)

neutral_network_file = f'{args.exp_file.split(".")[0]}_neutral_network_{args.walk_type}.pkl'
ne.pickle_neutral_network(neutral_network_file)


##### 

# signaling_dist, sig_sol = ne.longest_signaling_distance_from_original()
# neutral_walk_length, neutral_sol = ne.longest_neutral_walk_from_original()
# genotype_distance, genotype_sol = ne.longest_genotype_distance_from_original()

'''
# One parameter at a time
original_values = {}
param_data = {}
random_nonzero_indices = np.transpose(np.nonzero(best.state_genotype))
for i, indices in enumerate(random_nonzero_indices):
    original_values[i] = genotype_sol.state_genotype[*indices]

for mutate_param in range(23):
    ne = NeutralEngine(exp, genotype_sol, genotype_sol.state_genotype, mutate_layers=None, mutate_param=mutate_param)
    ne.run(3)
    param_data[mutate_param] = ne.param_data
param_data['original'] = original_values
print(param_data.keys())
print(len(param_data[0]))
with open(f'nw_Sep7_diamond_l2_t0.pkl', 'wb') as pf:
    pickle.dump(param_data, pf, protocol=pickle.HIGHEST_PROTOCOL)
'''




# print('Longest signaling distance: ', signaling_dist, f' (neutral path length: {sig_sol.neutral_counter})')
# print('Longest neutral path: ', neutral_walk_length, f' (signaling distance: {neutral_sol.signaling_distance})')
# print('Longest genotype distance: ', genotype_distance, f' (genotype distance: {genotype_sol.genotype_distance})')

# phenotypes_max_signaling = simulate_one_individual(exp, sig_sol)
# phenotypes_orig = simulate_one_individual(exp, ne.init_solution)
# phenotypes_max_neutral_walk = simulate_one_individual(exp, neutral_sol)
# phenotypes_max_genotype_distance = simulate_one_individual(exp, genotype_sol)


# visualize_all_layers(phenotypes_orig[0], './neutral_walk/orig.gif', base_layer_idx=exp.base_layer)
# visualize_all_layers(phenotypes_max_signaling[0], './neutral_walk/max_signaling.gif', base_layer_idx=exp.base_layer)
# visualize_all_layers(phenotypes_max_neutral_walk[0], './neutral_walk/max_neutralwalk.gif', base_layer_idx=exp.base_layer)
# visualize_all_layers(phenotypes_max_genotype_distance[0], './neutral_walk/genotype_dist.gif', base_layer_idx=exp.base_layer)

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
