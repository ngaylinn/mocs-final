import numpy as np
import matplotlib.pyplot as plt

from hillclimber import HillClimber
from simulation import NUM_STEPS, WORLD_SIZE, simulate

params = {
    'optimizer': 'hillclimber',
    'num_trials': 1,
    'target_population_size': 500,
    'max_generations': 10,
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

hc = HillClimber(params)
hc.evolve()
hc.pickle_hc('./test_hc.pkl')

print(hc.best_solution().fitness)