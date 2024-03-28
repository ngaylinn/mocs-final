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

####### SIMULATE SINGLE GENOTYPE #######
i = 12
init_phenotypes = afpo.make_seed_phenotypes(10)
state_genotypes_3 = state_genotypes.copy()
state_genotypes_3 = state_genotypes_3[:10]
phenotypes_3 = simulate(
    state_genotypes_3,
    afpo.n_layers,   
    afpo.population[0].around_start, 
    afpo.population[0].above_start,  
    init_phenotypes, 
    afpo.below_map, 
    afpo.above_map)


###### Evaluate phenotypic difference ######

# assert (state_genotypes_3[0] == state_genotypes[12]).all()
# print(state_genotypes.shape)
# print(phenotypes_1.shape, phenotypes_3.shape)
# if not (phenotypes_3[0] == phenotypes_1[i]).all():
#     print('Single different than multiple')

# Diff
diff = [np.sum(phenotypes_3[0, t] - phenotypes_2[i, t]) for t in range(NUM_STEPS)]
for i, d in enumerate(diff):
    print(i, d)

# (This evaluates the match between values of *all* grid cells at all timesteps) 
def different(phenotypes_1, phenotypes_2):
    n_different = 0
    sum_diff = 0
    max_diff = 0
    for i in range(len(phenotypes_2)):
        if not (phenotypes_2[i] == phenotypes_1[i]).all():
            n_different += 1
            diff = np.abs(phenotypes_2[i] - phenotypes_1[i])
            sum_diff += np.sum(diff)
            if np.max(diff) > max_diff:
                max_diff = np.max(diff)

    print('num diff: ', n_different)
    print('max diff: ', max_diff)

print(len(phenotypes_3), len(phenotypes_1))
different(phenotypes_1, phenotypes_2)
different(phenotypes_2, phenotypes_3)
different(phenotypes_1, phenotypes_3)
different(state_genotypes[:10], state_genotypes_3[:10])
# print(afpo.above_map)
# print(afpo.below_map)
