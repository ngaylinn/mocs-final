
from afpo import AgeFitnessPareto

experiment_parameters = {
    'target_population_size': 500,
    'max_generations': 2000,
}

if __name__ == '__main__':
    single_run = AgeFitnessPareto(experiment_parameters)
    single_run.evolve()