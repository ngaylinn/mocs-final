import numpy as np
import matplotlib.pyplot as plt

from afpo import AgeFitnessPareto, activation2int
from simulation import simulate

params = {
    'optimizer': 'hillclimber',
    'num_trials': 1,
    'target_population_size': 500,
    'max_generations': 50,
    'state_or_growth': None,
    'mutate_layers': None,
    'noise_rate': 0,
    'noise_intensity': 0,
    'sim_steps': 100,
    'shape': 'square',
    'layers': [
        {'res': 1},
        {'res': 2, 'base': True},
        {'res': 4},
        {'res': 8}
    ],
    'use_growth': True,
    'activation': 'sigmoid'
  }

afpo = AgeFitnessPareto(params)

# Initialize population
afpo.initialize_population()
pop_size = len(afpo.population)
# Get the population's genotypes for simulation
growth_genotypes, state_genotypes, unsimulated_indices = afpo.get_unsimulated_genotypes()
# Get the initial phenotype
init_phenotypes = afpo.make_seed_phenotypes(pop_size)

##### SIMULATE FIRST TIME ON GPU #####
print(f'Starting first {pop_size} simulations...')
phenotypes_1 = simulate(
    growth_genotypes, 
    state_genotypes, 
    afpo.n_layers, 
    afpo.base_layer,  
    afpo.population[0].around_start, 
    afpo.population[0].above_start, 
    afpo.use_growth, 
    init_phenotypes, 
    activation2int[afpo.activation])

fitness_scores_1 = afpo.evaluate_phenotypes(phenotypes_1)
for i, idx in enumerate(unsimulated_indices):
            afpo.population[idx].set_fitness(fitness_scores_1[i])
            afpo.population[idx].set_simulated(True)
            afpo.population[idx].set_phenotype(phenotypes_1[i][-1][afpo.base_layer] > 0)

# Reverse the order of the genotypes (should have no effect)
growth_genotypes_2 = growth_genotypes[::-1, :, :].copy()
state_genotypes_2 = state_genotypes[::-1, :, :].copy()
init_phenotypes = afpo.make_seed_phenotypes(pop_size)

##### SIMULATE SECOND TIME ON GPU #####
print(f'Starting second {pop_size} simulations...')
phenotypes_2 = simulate(
    growth_genotypes_2, 
    state_genotypes_2, 
    afpo.n_layers, 
    afpo.base_layer,  
    afpo.population[0].around_start, 
    afpo.population[0].above_start, 
    afpo.use_growth, 
    init_phenotypes, 
    activation2int[afpo.activation])

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
for i in range(len(fitness_scores_2)):
    if not (phenotypes_2[i] == phenotypes_1[i]).all():
        n_different += 1
        ###### Plot divergence of values #########
        # phenotypes_1[i] phenotypes_2[i]

        # divergence = [np.sum(phenotypes_1[i, t] - phenotypes_2[i,t]) for t in range(params['sim_steps'])]

        # print(state_genotypes[i], state_genotypes_2[i])
        # print(phenotypes_1[i,2,afpo.base_layer,::8, ::8])
        # print(phenotypes_2[i,2,afpo.base_layer,::8, ::8])
        # print(afpo.population[0].above_start)
        # print(afpo.population[0].around_start)

        # plt.plot(range(params['sim_steps']), divergence)
        # plt.show()
        # exit()
    else:
        print((fitness_scores_1[i], fitness_scores_2[i]))

        if fitness_scores_2[i] != 2048 and fitness_scores_2[i] != 3072:
              print(fitness_scores_2[i])
              print(phenotypes_2[i,-1,afpo.base_layer,::8, ::8])
              print(phenotypes_1[i,-1,afpo.base_layer,::8, ::8])
            #   exit()



print('num diff phenotype: ', n_different)

n_different = 0
for i in range(len(fitness_scores_2)):
    if not (fitness_scores_1[i] == fitness_scores_2[i]):
        n_different += 1
        
print('num diff fitness: ', n_different)
