import os
import matplotlib.pyplot as plt
import pickle
import time
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

    return phenotypes

parser = argparse.ArgumentParser()
parser.add_argument('exp_file', type=str, help='Experiment File')
args = parser.parse_args()

exp = get_experiment(args.exp_file)
best = exp.best_solution()

best.neutral_counter = 0
best.genotype_distance = 0
print('best fitness: ', best.fitness)

# solution_genome_size = len(np.transpose(np.nonzero(best.state_genotype)))
# neutral_percent_sum = 0
# for j in range(solution_genome_size):
#     ne = NeutralEngine(exp, best, best.state_genotype, mutate_layers=None, mutate_param=j)
#     ne.run(1)
#     neutral_percent_sum += ne.neutral_percent
#     print(f'Neutral percent: {ne.neutral_percent}')

# neutral_percent = neutral_percent_sum / solution_genome_size
# print(f'Neutral neighborhood percentage: {neutral_percent}')

# exit(1)


for i in range(5): 
    print(f'Iteration {i}')
    start = time.time()
    # Take a far neutral walk in genotype space
    best.genotype_distance = 0
    ne = NeutralEngine(exp, best, best.state_genotype, mutate_layers=None)
    ne.run(2)

    ##### Grab just the furthest neutral walk in genotype space
    genotype_distance, genotype_sol = ne.longest_genotype_distance_from_original()
    # neutral_counter, neutral_sol = ne.longest_neutral_walk_from_original()

    solution_genome_size = len(np.transpose(np.nonzero(genotype_sol.state_genotype)))
    # solution_genome_size = len(np.transpose(np.nonzero(neutral_sol.state_genotype)))

    print(f'Genotype distance: {genotype_distance}')
    # print(f'Neutral counter: {neutral_counter}')

    ##### Now measure that genotype's neutral neighborhood percentage
    neutral_percent_sum = 0
    for j in range(solution_genome_size):
        ne = NeutralEngine(exp, genotype_sol, genotype_sol.state_genotype, mutate_layers=None, mutate_param=j)
        # ne = NeutralEngine(exp, neutral_sol, neutral_sol.state_genotype, mutate_layers=None, mutate_param=j)
        ne.run(1)
        neutral_percent_sum += ne.neutral_percent

    neutral_percent = neutral_percent_sum / solution_genome_size

    with open(f'robustness_experiment/neutral_walk_genotype_distance_{genotype_distance}_percent_{neutral_percent}.pkl', 'wb') as pf:
        pickle.dump(genotype_sol, pf, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(f'robustness_experiment/neutral_walk_neutral_steps_{neutral_counter}_percent_{neutral_percent}.pkl', 'wb') as pf:
    #     pickle.dump(neutral_sol, pf, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Neutral neighborhood percentage: {neutral_percent_sum}')
    end = time.time()

    print(f'[Iteration {i}] Time taken: {end - start}s')





print('Longest signaling distance: ', signaling_dist, f' (neutral path length: {sig_sol.neutral_counter})')
print('Longest neutral path: ', neutral_walk_length, f' (signaling distance: {neutral_sol.signaling_distance})')
print('Longest genotype distance: ', genotype_distance, f' (genotype distance: {genotype_sol.genotype_distance})')

phenotypes_max_signaling = simulate_one_individual(exp, sig_sol)
phenotypes_orig = simulate_one_individual(exp, ne.init_solution)
phenotypes_max_neutral_walk = simulate_one_individual(exp, neutral_sol)
phenotypes_max_genotype_distance = simulate_one_individual(exp, genotype_sol)


visualize_all_layers(phenotypes_orig[0], './neutral_walk/orig.gif', base_layer_idx=exp.base_layer)
visualize_all_layers(phenotypes_max_signaling[0], './neutral_walk/max_signaling.gif', base_layer_idx=exp.base_layer)
visualize_all_layers(phenotypes_max_neutral_walk[0], './neutral_walk/max_neutralwalk.gif', base_layer_idx=exp.base_layer)
visualize_all_layers(phenotypes_max_genotype_distance[0], './neutral_walk/genotype_dist.gif', base_layer_idx=exp.base_layer)
