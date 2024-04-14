import pickle
import numpy as np
from simulation import simulate

with open('./experiments/Apr13_complex_up_nonlocal/complex_l4_b1/hc_2000.pkl', 'rb') as pf:
    hc = pickle.load(pf)

best = hc.best_solution()

below_maps = np.array([best.below_map])

init_phenotypes = hc.make_seed_phenotypes(1)

phenotypes = simulate(
    np.array([best.state_genotype]),
    4,
    best.around_start,
    init_phenotypes,
    below_maps
)

fitnesses, _ = hc.evaluate_phenotypes(phenotypes)

print(best.fitness)
print(fitnesses)

