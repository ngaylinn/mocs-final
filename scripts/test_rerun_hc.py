

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hillclimber import HillClimber
from simulation import NUM_STEPS, WORLD_SIZE, simulate

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

with open(args.file, 'rb') as pf:
    hc = pickle.load(pf)

# with open('genotypes.pkl', 'rb') as pf:
#     genotypes = pickle.load(pf)

best_solution = hc.best_solution()

print('Top fitness: ', best_solution.fitness)
print('Population: ', len(hc.parent_population), ' individuals')

# All genotypes
genotypes = np.array([sol.state_genotype for k, sol in hc.parent_population.items()], dtype=np.float32)
init_phenotypes = hc.make_seed_phenotypes(len(genotypes))

# genotype_1 = np.array([best_solution.state_genotype], dtype=np.float32)
# init_phenotypes_1 = hc.make_seed_phenotypes(1)
rand_id = list(hc.parent_population.keys())[0]

phenotypes_3 = simulate(
    genotypes,
    hc.n_layers,   
    hc.parent_population[rand_id].around_start, 
    hc.parent_population[rand_id].above_start,  
    init_phenotypes, 
    hc.below_map, 
    hc.above_map)

fitnesses = hc.evaluate_phenotypes(phenotypes_3)

phenotype = phenotypes_3[0][-1][hc.base_layer] > 0

# print('Phenotypes match: ', (phenotype == best_solution.phenotype).all())
# print(sum(best_solution.phenotype))
print(min(fitnesses))
# print(fitnesses)
