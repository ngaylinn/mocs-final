import pickle
import numpy as np
import argparse

from simulation import simulate

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

with open(args.file, 'rb') as pf:
    afpo = pickle.load(pf)

best_solution = afpo.best_solution()
print(best_solution.fitness)
print(best_solution.been_simulated)
print(best_solution.state_genotype)

genotype = best_solution.state_genotype
init_phenotypes = afpo.make_seed_phenotypes(1)

phenotypes = simulate(
    np.array([genotype]), 
    afpo.n_layers,   
    afpo.population[0].around_start, 
    afpo.population[0].above_start,  
    init_phenotypes, 
    afpo.below_map, 
    afpo.above_map)

fitness = afpo.evaluate_phenotypes(phenotypes)

print(fitness)

print(phenotypes[0][-1][afpo.base_layer] > 0)