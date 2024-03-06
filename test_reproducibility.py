import time
from afpo import AgeFitnessPareto, activation2int
from hillclimber import HillClimber
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
        {'res': 2},
        {'res': 4},
        {'res': 8, 'base': True}
    ],
    'use_growth': True,
    'activation': 'sigmoid'
  }

afpo = AgeFitnessPareto(params)

afpo.initialize_population()
unsimulated_growth_genotypes, unsimulated_state_genotypes, unsimulated_indices = afpo.get_unsimulated_genotypes()
init_phenotypes = afpo.make_seed_phenotypes(unsimulated_growth_genotypes.shape[0])
noise = afpo.generate_noise()

##### SIMULATE ON GPUs #####
print(f'Starting {afpo.target_population_size} simulations...')
phenotypes_1 = simulate(
    unsimulated_growth_genotypes, 
    unsimulated_state_genotypes, 
    afpo.n_layers, 
    afpo.base_layer,  
    afpo.population[0].around_start, 
    afpo.population[0].above_start, 
    afpo.use_growth, 
    init_phenotypes, 
    activation2int[afpo.activation],
    noise)

fitness_scores_1 = afpo.evaluate_phenotypes(phenotypes_1)
for i, idx in enumerate(unsimulated_indices):
            afpo.population[idx].set_fitness(fitness_scores_1[i])
            afpo.population[idx].set_simulated(True)
            afpo.population[idx].set_phenotype(phenotypes_1[i][-1][afpo.base_layer] > 0)

new_noise = afpo.generate_noise()

phenotypes_2 = simulate(
    unsimulated_growth_genotypes, 
    unsimulated_state_genotypes, 
    afpo.n_layers, 
    afpo.base_layer,  
    afpo.population[0].around_start, 
    afpo.population[0].above_start, 
    afpo.use_growth, 
    init_phenotypes, 
    activation2int[afpo.activation],
    new_noise)

fitness_scores_2 = afpo.evaluate_phenotypes(phenotypes_2)
for i, idx in enumerate(unsimulated_indices):
            afpo.population[idx].set_fitness(fitness_scores_2[i])
            afpo.population[idx].set_simulated(True)
            afpo.population[idx].set_phenotype(phenotypes_2[i][-1][afpo.base_layer] > 0)

n_different = 0
for i in range(len(fitness_scores_2)):
    if not (phenotypes_2[i] == phenotypes_1[i]).all():
        n_different += 1

print('num diff: ', n_different)
# print(fitness_scores_1)
# print(fitness_scores_2)