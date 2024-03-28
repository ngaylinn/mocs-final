import numpy as np
import matplotlib.pyplot as plt

from afpo import AgeFitnessPareto, activation2int
from simulation import NUM_STEPS, WORLD_SIZE, simulate

params = {
    'optimizer': 'afpo',
    'num_trials': 1,
    'target_population_size': 500,
    'max_generations': 1,
    'mutate_layers': None,
    'neighbor_map_type': 'spatial',
    'n_random_individuals_per_generation': 100,
    'noise_rate': 0,
    'noise_intensity': 0,
    'sim_steps': 100,
    'shape': 'square',
    'layers': [
        {'res': 1, 'base': True},
        {'res': 2},
        {'res': 4},
        {'res': 8}
    ],
    'activation': 'sigmoid'
  }

afpo = AgeFitnessPareto(params)
afpo.evolve()
afpo.pickle_afpo('./test_afpo.pkl')

print(afpo.best_solution().fitness)