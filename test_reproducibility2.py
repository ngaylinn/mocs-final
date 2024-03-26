import numpy as np
import matplotlib.pyplot as plt

from afpo import AgeFitnessPareto, activation2int
from simulation import NUM_STEPS, WORLD_SIZE, simulate

params = {
    'optimizer': 'hillclimber',
    'num_trials': 1,
    'target_population_size': 100,
    'max_generations': 3,
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

afpo.evolve()

best_solution = afpo.best_solution()