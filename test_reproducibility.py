import numpy as np
import matplotlib.pyplot as plt

from afpo import AgeFitnessPareto, activation2int
from simulation import NUM_STEPS, WORLD_SIZE, simulate

params = {
    'optimizer': 'hillclimber',
    'num_trials': 1,
    'target_population_size': 500,
    'max_generations': 50,
    'mutate_layers': None,
    'neighbor_map_type': 'random',
    'n_random_individuals_per_generation': 100,
    'noise_rate': 0,
    'noise_intensity': 0,
    'sim_steps': 100,
    'shape': 'square',
    'layers': [
        {'res': 1},
        {'res': 2},
        {'res': 4},
        {'res': 8, 'base': True}
    ],
    'activation': 'sigmoid'
  }

afpo = AgeFitnessPareto(params)

# Initialize population
afpo.initialize_population()
pop_size = len(afpo.population)
# Get the population's genotypes for simulation
state_genotypes, unsimulated_indices = afpo.get_unsimulated_genotypes()
# Get the initial phenotype
init_phenotypes = afpo.make_seed_phenotypes(pop_size)

##### SIMULATE FIRST TIME ON GPU #####
print(f'Starting first {pop_size} simulations...')
phenotypes_1 = simulate(
    state_genotypes, 
    afpo.n_layers, 
    afpo.population[0].around_start, 
    afpo.population[0].above_start,  
    init_phenotypes, 
    activation2int[afpo.activation],
    afpo.below_map,
    afpo.above_map)

fitness_scores_1 = afpo.evaluate_phenotypes(phenotypes_1)
for i, idx in enumerate(unsimulated_indices):
            afpo.population[idx].set_fitness(fitness_scores_1[i])
            afpo.population[idx].set_simulated(True)
            afpo.population[idx].set_phenotype(phenotypes_1[i][-1][afpo.base_layer] > 0)

# Reverse the order of the genotypes (should have no effect)
state_genotypes_2 = state_genotypes[::-1, :, :].copy()
init_phenotypes = afpo.make_seed_phenotypes(pop_size)

##### SIMULATE SECOND TIME ON GPU #####
print(f'Starting second {pop_size} simulations...')
phenotypes_2 = simulate(
    state_genotypes_2, 
    afpo.n_layers,   
    afpo.population[0].around_start, 
    afpo.population[0].above_start,  
    init_phenotypes, 
    activation2int[afpo.activation],
    afpo.below_map, 
    afpo.above_map)

# Reverse the output...
phenotypes_2 = phenotypes_2[::-1, :, :, :, :]
print('Shape: ', phenotypes_2.shape)

fitness_scores_2 = afpo.evaluate_phenotypes(phenotypes_2)
for i, idx in enumerate(unsimulated_indices):
            afpo.population[idx].set_fitness(fitness_scores_2[i])
            afpo.population[idx].set_simulated(True)
            afpo.population[idx].set_phenotype(phenotypes_2[i][-1][afpo.base_layer] > 0)



###### Evaluate phenotypic difference ######
# (This evaluates the match between values of *all* grid cells at all timesteps) 
n_different = 0
sum_diff = 0
max_diff = 0
for i in range(len(fitness_scores_2)):
    if not (phenotypes_2[i] == phenotypes_1[i]).all():
        n_different += 1
        diff = np.abs(phenotypes_2[i] - phenotypes_1[i])
        sum_diff += np.sum(diff)
        if np.max(diff) > max_diff:
            max_diff = np.max(diff)

print('num diff: ', n_different)
# print('avg diff: ', sum_diff / n_different)
print('max diff: ', max_diff)

print(afpo.above_map)
print(afpo.below_map)
