import numpy as np
import pickle

from hillclimber import HillClimber
from simulation import simulate, make_seed_phenotypes
from afpo import Solution
from vis import visualize_all_layers

layers = [
{'res': 1, 'base': True},
{'res': 1}
]

with open('experiments/exp_circle_2layer_control/circle_l2_local_control/hc_2000.pkl', 'rb') as pf:
    hc = pickle.load(pf)

sol = hc.best_solution()

genotypes = np.array([sol.state_genotype])
phenotypes = make_seed_phenotypes(1, n_layers=2)


out_phenotypes = simulate(
  genotypes,
  2,
  phenotypes
)

print(out_phenotypes.shape)
visualize_all_layers(out_phenotypes[0], 'out.gif', base_layer_idx=0)

with open('./sol_phenotype.pkl', 'wb') as pf:
    pickle.dump(out_phenotypes[0], pf)
