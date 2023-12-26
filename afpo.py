import functools
import time
import pickle

import numpy as np

from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH

activation2int = {
    'sigmoid': ACTIVATION_SIGMOID,
    'tanh': ACTIVATION_TANH,
    'relu': ACTIVATION_RELU
}

@functools.total_ordering # Sortable by fitness
class Solution:
    def __init__(self, n_layers=3):
        self.n_layers = n_layers
        self.age = 0
        self.been_simulated = False
        self.fitness = None
        self.phenotype = None
        self.genotype = self.randomize_genome()

    def make_offspring(self):
        child = Solution(n_layers=self.n_layers)
        child.genotype = child.genotype.copy()
        child.mutate()
        return child

    def increment_age(self):
        self.age += 1

    def set_phenotype(self, phenotype):
        self.phenotype = phenotype

    def set_simulated(self, new_simulated):
        self.been_simulated = new_simulated

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def mutate(self):
        random_layer = np.random.choice(range(NUM_LAYERS), p=[np.sum(l != 0)/self.total_weights for l in self.genotype])
        random_nonzero_indices = np.transpose(np.nonzero(self.genotype[random_layer]))
        r, c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))]

        self.genotype[random_layer, r, c] = np.random.random() * 2 - 1

    def dominates(self, other):
        return all([self.age <= other.age, self.fitness <= other.fitness])

    def __eq__(self, other):
        return all([
            other is not None,
            isinstance(other, self.__class__),
            self.fitness == other.fitness
        ])

    def __lt__(self, other):
        return all([
            other is not None,
            isinstance(other, self.__class__),
            self.fitness < other.fitness
        ])

    def randomize_genome(self):
        # Randomly initialize the NN weights (3 layers, input neurons, output neurons)
        genotype = np.random.random((NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)).astype(np.float32) * 2 - 1
        # Mask weights
        for l in range(NUM_LAYERS):
            if l < self.n_layers:
                genotype[l] *= get_layer_mask(l)
            else:
                genotype[l] *= np.zeros(genotype[l].shape)
        
        self.total_weights = np.sum(genotype != 0)
        return genotype

class AgeFitnessPareto:
    def __init__(self, experiment_constants):
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.use_growth = experiment_constants['use_growth']
        self.activation = experiment_constants['activation']
        self.fitness_threshold = experiment_constants['fitness_threshold']
        self.population = []
        self.current_generation = 1

        self.best_fitness_history = []

    def evolve(self):
        self.initialize_population()
        while self.current_generation <= self.max_generations:
            print(f'Generation {self.current_generation}')
            self.evolve_one_generation()
            self.current_generation += 1

        return self.best_solution()

    def evolve_one_generation(self):
        # Actually run the simulations, and time how long it takes.
        start = time.perf_counter()

        init_phenotypes = self.make_seed_phenotypes()
        unsimulated_genotypes, unsimulated_indices = self.get_unsimulated_genotypes()
        
        ##### SIMULATE ON GPUs #####
        print(f'Starting {self.target_population_size} simulations...')
        phenotypes = simulate(
            unsimulated_genotypes, self.layers, self.use_growth, init_phenotypes, activation2int[self.activation])

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        fitness_scores = self.evaluate_phenotypes(phenotypes)
        # Set the fitness and simulated flag for each of the just-evaluated solutions
        for i, idx in enumerate(unsimulated_indices):
            self.population[idx].set_fitness(fitness_scores[i])
            self.population[idx].set_simulated(True)
            # self.population[idx].set_phenotype(phenotypes[i])

        print('Average fitness:',
              np.mean([sol.fitness for sol in self.population]))
        print('Average age:',
              np.mean([sol.age for sol in self.population]))
        # Reduce the population
        self.reduce_population()
        # Increment ages by 1
        for sol in self.population:
            sol.increment_age()
        # Extend the population using tournament selection
        self.extend_population()

        self.best_fitness_history.append(self.best_solution())


    def initialize_population(self):
        # Initialize target_population_size random solutions
        self.population = [
            Solution(n_layers=self.layers) for _ in range(self.target_population_size)
        ]

    def extend_population(self):
        new_individuals = []
        # 1 - Breed: do tournament selection
        # The minus one is to make room for one random individual at the end.
        for _ in range(self.target_population_size - 1):
            # Randomly select an individual using tournament selection
            parent = self.tournament_select()
            new_individuals.append(parent.make_offspring())

        self.population += new_individuals

        # Add a single random individual
        self.population.append(Solution(n_layers=self.layers))

    def reduce_population(self):
        # Remove individuals until target population is reached
        while len(self.population) > self.target_population_size:
            sol1, sol2 = np.random.choice(self.population, 2, replace=False)
            # Note that it's possible that NEITHER dominates the other.
            if sol1.dominates(sol2):
                self.population.remove(sol2)
            elif sol2.dominates(sol1):
                self.population.remove(sol1)

    def tournament_select(self):
        """
        Tournament selection randomly chooses two individuals from the population and
        selects the better (based on a primary objective) of the two for reproduction/mutation
        """
        sol1, sol2 = np.random.choice(self.population, 2, replace=False)
        return min(sol1, sol2)


    def get_unsimulated_genotypes(self):
        # Filter out just the genotypes that haven't been simulated yet.
        unsimulated_genotypes = [
            sol.genotype for sol in self.population if not sol.been_simulated
        ]
        unsimulated_indices = [
            i for i, sol in enumerate(self.population) if not sol.been_simulated
        ]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(unsimulated_genotypes, dtype=np.float32), unsimulated_indices


    def evaluate_phenotypes(self, phenotypes):
        """Score a set of phenotypes generated by the simulate function."""
        target = np.full((WORLD_SIZE, WORLD_SIZE), DEAD)
        target[(WORLD_SIZE // 4):(WORLD_SIZE//4 * 3), (WORLD_SIZE // 4):(WORLD_SIZE//4 * 3)] = ALIVE

        # Infer pop_size from phenotypes
        pop_size = phenotypes.shape[0]
        # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
        assert phenotypes.shape == (pop_size, NUM_STEPS, NUM_LAYERS, WORLD_SIZE, WORLD_SIZE)
        assert target.shape == (WORLD_SIZE, WORLD_SIZE)

        # Allocate space for results.
        fitness_scores = np.zeros(pop_size, dtype=np.uint32)

        # For each individual in the population...
        for i in range(pop_size):
            # Look at just the final state of the layer0 part of the phenotype.
            # Compare it to the target image, and sum up the deltas to get the
            # final fitness score (lower is better, 0 is a perfect score).
            fitness_scores[i] = np.sum(np.abs(target - (phenotypes[i][-1][0] > self.fitness_threshold)))

        return fitness_scores


    # @functools.cache
    def make_seed_phenotypes(self):
        """Starting phenotypes to use by default (one ALIVE cell in middle)."""
        # For each inidividual, capture phenotype development over NUM_STEPS. Each
        # phenotype has NUM_LAYERS layers which are all WORLD_SIZE x WORLD_SIZE
        # squares. Layer0 is the "ground truth" of the CA while layers 1 and 2
        # represent a sort of hierarchical internal state for the organism. Layers
        # 1 and 2 are conceptually smaller than layer0 (1/4 and 1/8 respectively),
        # but are represented using arrays of the same size for simplicity.
        phenotypes = np.full(
            (self.target_population_size, NUM_STEPS, NUM_LAYERS, WORLD_SIZE, WORLD_SIZE),
            DEAD, dtype=np.float32)

        # Use a single ALIVE pixel in the middle of the CA world as the initial
        # phenotype state for all individuals in the population.
        for i in range(self.target_population_size):
            middle = WORLD_SIZE // 2
            phenotypes[i][0][0][middle][middle] = ALIVE

        return phenotypes


    def pickle_afpo(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)


    def best_solution(self):
        return min([sol for sol in self.population if sol.fitness is not None])
