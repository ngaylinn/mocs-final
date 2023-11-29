import copy
import functools
import time

import numpy as np

from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS

@functools.total_ordering # Sortable by fitness
class Solution:
    def __init__(self):
        self.age = 0
        self.been_simulated = False
        self.fitness = None
        self.phenotype = None
        self.randomize_genome()

    def make_offspring(self):
        child = copy.deepcopy(self)
        child.mutate()
        child.age = 0

    def increment_age(self):
        self.age += 1

    def set_simulated(self):
        self.been_simulated = True

    def mutate(self):
        pass

    def dominates(self, other):
        return all([self.age < other.age, self.fitness < other.fitness])

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
        self.genotype = np.random.random((3, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)).astype(np.float32) * 2 - 1


class AgeFitnessPareto:
    def __init__(self, experiment_constants):
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.population = []

    def evolve(self):
        self.initialize_population()
        for _ in range(self.max_generations):
            self.evolve_one_generation()
        return max(self.population)

    def evolve_one_generation(self):
        # Increment ages by 1
        for sol in self.population:
            sol.increment_age()
        # Extend the population using tournament selection
        self.extend_population()

        # Actually run the simulations, and time how long it takes.
        print(f'Starting {self.target_population_size} simulations...')
        start = time.perf_counter()

        init_phenotypes = self.make_seed_phenotypes()
        unsimulated_genotypes = self.get_unsimulated_genotypes()
        layers = [self.layers for _ in range(len(unsimulated_genotypes))]
        phenotypes = simulate(unsimulated_genotypes, layers, init_phenotypes)

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        # Evaluate the phenotypes
        self.evaluate_phenotypes(phenotypes)

        # Make sure every individual has been set to simulated
        for sol in self.population:
            sol.set_simulated()

        # Reduce the population
        self.reduce_population()

    def initialize_population(self):
        # Initialize target_population_size random solutions
        self.population = [
            Solution() for _ in range(self.target_population_size)
        ]

    def extend_population(self):
        # 1 - Breed: do tournament selection
        for _ in range(self.target_population_size):
            # Randomly select an individual using tournament selection
            parent = self.tournament_select()
            self.population.append(parent.make_offspring())

        # Add a single random individual
        self.population.append(Solution())

    def reduce_population(self):
        # Remove individuals until target population is reached
        while len(self.population) > self.target_population_size:
            sol1, sol2 = np.random.choice(self.population, 2, replace=False)
            # Note that it's possible that NEITHER dominates the other.
            if sol1.dominates(sol2):
                self.population.remove(sol1)
            elif sol2.dominates(sol1):
                self.population.remove(sol2)

    def tournament_select(self):
        """
        Tournament selection randomly chooses two individuals from the population and
        selects the better (based on a primary objective) of the two for reproduction/mutation
        """
        sol1, sol2 = np.random.choice(self.population, 2, replace=False)
        return max(sol1, sol2)

    def evaluate_phenotypes(self, phenotypes):
        for sol, phenotype in zip(population, phenotypes):
            fitness = ... # TODO: Evaluate fitness!
            sol.fitness = fitness
            sol.phenotype = phenotype

    def get_unsimulated_genotypes(self):
        # Filter out just the genotypes that haven't been simulated yet.
        unsimulated_genotypes = [
            sol.genotype for sol in self.population if not sol.been_simulated
        ]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(unsimulated_genotypes, dtype=np.float32)

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
