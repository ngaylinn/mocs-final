from pathlib import Path
import tracemalloc
import linecache
import os
import time
import argparse

import numpy as np

from afpo import Solution
from hillclimber import HillClimber
from simulation import simulate, make_seed_phenotypes, visualize, generate_random_below_map, do_they_percolate


def get_below_maps(n=500):
    n_layers = 4
    below_maps = np.zeros((500, 4, 4, 3))

    n_generated = 0
    
    # Generate random below maps
    prospective_below_maps = np.array([generate_random_below_map() for _ in range(n)])

    sol = Solution(layers=[{'res': 1}, {'res': 2}, {'res': 4}, {'res': 8, 'base': True}], id=0, below_map=prospective_below_maps[0])

    while n_generated < 500:
        # Get initial phenotypes
        seed_phenotypes = make_seed_phenotypes(500)

        # All ones genotypes to allow percolation...
        genotypes = np.ones((n, 4, 14))

        phenotypes = simulate(
                genotypes,
                sol.n_layers,
                sol.around_start,
                seed_phenotypes, 
                prospective_below_maps)
        
        percolators = do_they_percolate(phenotypes)
        n_percolators = sum(percolators)

        percolator_below_maps = prospective_below_maps[percolators]
        
        print(len(percolator_below_maps))

        for i in range(n_percolators):
            below_maps[n_generated] = percolator_below_maps[i]
            n_generated += 1
            if n_generated == 500:
                break
        
    return below_maps
        
