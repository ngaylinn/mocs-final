

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from afpo import AgeFitnessPareto, activation2int
from simulation import NUM_STEPS, WORLD_SIZE, simulate

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

with open(args.file, 'rb') as pf:
    afpo = pickle.load(pf)

best_solution = afpo.best_solution()

print('Top fitness: ', best_solution.fitness)
print('Population: ', len(afpo.population), ' individuals')

last_gen_solutions = [sol for sol in afpo.population if sol.been_simulated]

# All genotypes
genotypes = np.array([sol.state_genotype for sol in last_gen_solutions], dtype=np.float32)
init_phenotypes = afpo.make_seed_phenotypes(len(last_gen_solutions))

# genotype_1 = np.array([best_solution.state_genotype], dtype=np.float32)
# init_phenotypes_1 = afpo.make_seed_phenotypes(1)

print(len(last_gen_solutions))

phenotypes_3 = simulate(
    genotypes,
    afpo.n_layers,   
    afpo.population[0].around_start, 
    afpo.population[0].above_start,  
    init_phenotypes, 
    afpo.below_map, 
    afpo.above_map)

fitnesses = afpo.evaluate_phenotypes(phenotypes_3)

phenotype = phenotypes_3[0][-1][afpo.base_layer] > 0

# print('Phenotypes match: ', (phenotype == best_solution.phenotype).all())
# print(sum(best_solution.phenotype))
print(min(fitnesses))
