import numpy as np
import pickle
import argparse

from hillclimber import HillClimber
from simulation import simulate, make_seed_phenotypes
from afpo import Solution
from vis import visualize_all_layers

params = {
  'target_population_size': 500,
  'max_generations': 50,
  'layers': [{'res': 1, 'base': True}, {'res': 1}],
  'activation': 'sigmoid',
  'shape': 'circle',
  'mutate_layers': None,
  'neighbor_map_type': 'spatial'
}

hc = HillClimber(params)
hc.evolve()

hc.pickle_hc('hc.pkl')

best = hc.best_solution()

genotypes = np.array([best.state_genotype])
phenotypes = make_seed_phenotypes(1, n_layers=2)

out_phenotypes = simulate(
  genotypes,
  2,
  phenotypes
)

with open('hc_best.pkl', 'wb') as pf:
    pickle.dump(out_phenotypes[0], pf)
