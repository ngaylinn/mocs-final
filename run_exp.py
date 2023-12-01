from pathlib import Path
import tracemalloc
import linecache
import os
import time

import numpy as np

from afpo import AgeFitnessPareto
from simulation import visualize

experiment_parameters = {
    'num_trials': 1,
    'target_population_size': 50,
    'max_generations': 20,
    'layers': 3,
    'use_growth': True,
}

def main():
    # Delete all the visualizations generated in the last run (if any exist)
    for file in Path().glob('*.gif'):
        file.unlink()

    # Try running with 0 extra layers (control, traditional CA) or with 2 extra
    # layers (hierarchical CA)
    # TODO: Do we also want to try with and without growth enabled?
    for arm in ('Control', 'Experiment'):
        layers = 1 if arm == 'Control' else 3
        fitness_scores = []
        # Run a few instances of each
        for trial in range(experiment_parameters['num_trials']):
            experiment_parameters['layers'] = layers
            # TODO: It would be more efficient to run all trials for both
            # experiment and control in one batch, but that would also take
            # some non-trivial refactoring. We should add that optimization
            # only if necessary.
            single_run = AgeFitnessPareto(experiment_parameters)
            sol = single_run.evolve()

            single_run.pickle_afpo(f'{arm}_t{trial}.pkl')

            fitness_scores.append(sol.fitness)
            # visualize(
            #     sol.phenotype[0], # Save just the first layer
            #     f'{arm}_t{trial}_f{sol.fitness}.gif')

        print(f'Mean fitness for {arm}: {np.mean(fitness_scores)}')

if __name__ == '__main__':
    main()
