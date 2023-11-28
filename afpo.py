import pickle
import copy
import time

import numpy as np

from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS

class Solution:
    def __init__(self, id):
        self.id = id
        self.age = 0
        self.been_simulated = False
        self.fitness = None

        self.initialize_genome()

    def set_id(self, new_id):
        self.id = new_id
    
    def reset_age(self):
        self.age = 0

    def get_age(self):
        return self.age

    def increment_age(self):
        self.age += 1

    def set_simulated(self, new_simulated):
        self.been_simulated = new_simulated

    def get_fitness(self):
        return self.fitness

    def mutate(self):
        pass

    def dominates_other(self, other_solution):
        return all([self.age < other_solution.get_age(), self.fitness < other_solution.get_fitness()])

    def initialize_genome(self):
        # Randomly initialize the NN weights (3 layers, input neurons, output neurons)
        self.genotype = np.random.random((3, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)).astype(np.float32) * 2 - 1



class AgeFitnessPareto:
    def __init__(self, experiment_constants):
        self.current_generation = 0
        self.population = {}

        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']


    def evolve(self):
        # Initialize population if the current generation is 0
        if self.current_generation == 0:
            self.initialize_population()

        # Evolve generations one at a time
        while self.current_generation <= self.max_generations:
            self.evolve_one_generation()
            self.current_generation += 1


    def evolve_one_generation(self):
        # Increment ages by 1
        self.increment_ages()
        # Extend the population using tournament selection
        self.extend_population()

        # Actually run the simulations, and time how long it takes.
        print(f'Starting {self.target_population_size} simulations...')
        start = time.perf_counter()

        init_phenotypes = self.make_seed_phenotypes()
        unsimulated_genotypes = self.get_unsimulated_genotypes()
        layers = [2 for _ in range(len(unsimulated_genotypes))]                 # TODO: handle layers within Solution class and get_unsimulated_genotypes()
        phenotypes = simulate(unsimulated_genotypes, layers, init_phenotypes)

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        # Evaluate the phenotypes
        self.evaluate_phenotypes(phenotypes)

        # Make sure every individual has been set to simulated
        for sol in self.population:
            sol.set_simulated(True)

        # Reduce the population 
        self.reduce_population()


    def initialize_population(self):
        # Initialize target_population_size random solutions
        for _ in range(self.target_population_size):
            random_id = hash(np.random.random())
            self.population[random_id] = Solution(random_id)


    def extend_population(self):
        # 1 - Breed: do tournament selection 
        for _ in range(self.target_population_size):
            # Randomly select an individual using tournament selection
            parent = self.tournament_select()

            # Child is a deepcopy of the parent 
            child = copy.deepcopy(parent)

            # Mutate the child
            child.mutate()

            # Add the child to the population
            random_id = hash(np.random.random())
            child.set_id(random_id)
            child.reset_age()
            self.population[random_id] = child

        # Add a single random individual 
        random_id = hash(np.random.random())
        self.population[random_id] = Solution(random_id)


    def reduce_population(self):
        # Remove individuals until target population is reached
        while len(self.population) > self.target_population_size:
            i1 = np.random.choice(list(self.population.keys()))
            i2 = np.random.choice(list(self.population.keys()))
            while i2 == i1:
                i2 = np.random.choice(list(self.population.keys()))
            if self.dominates(i1, i2): # i1 dominates
                self.population.pop(i2)
            elif self.dominates(i2, i1): # i2 dominates
                self.population.pop(i1)

    
    def dominates(self, i1, i2):
        return self.population[i1].dominates_other(self.population[i2])


    def tournament_select(self):
        """ 
        Tournament selection randomly chooses two individuals from the population and 
        selects the better (based on a primary objective) of the two for reproduction/mutation
        """
        p1 = np.random.choice(list(self.population.keys()))
        p2 = np.random.choice(list(self.population.keys()))
        while p2 == p1:
            p2 = np.random.choice(list(self.population.keys()))

        # Tournament over the primary objective 
        if self.population[p1].get_fitness() > self.population[p2].get_fitness():
            return p1
        else:
            return p2


    def evaluate_phenotypes(self, phenotypes):
        pass


    def get_unsimulated_genotypes(self):
        # Filter out the unsimulated solutions
        unsimulated_solutions = [self.population[id] for id in self.population if not sol.been_simulated]

        # Aggregate the genotypes into a single matrix for simulation
        genotypes = np.random.random((len(unsimulated_solutions), 3, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)).astype(np.float32) * 2 - 1
        for i, sol in enumerate(unsimulated_solutions):
            genotypes[i] = sol.genotype

        return genotypes


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


    def increment_ages(self):
        for individual in self.population:
            individual.increment_age()

    