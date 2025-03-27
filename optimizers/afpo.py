import functools
import time
import pickle
from collections import Counter

import numpy as np

from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS
from util import create_hollow_circle, create_square, create_diamond, create_plus, create_complex, simulate_one_individual
from solution import Solution

class AgeFitnessPareto:
    def __init__(self, experiment_constants):
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.activation = experiment_constants['activation']
        self.shape = experiment_constants['shape']
        self.neighbor_map_type = experiment_constants['neighbor_map_type'] # 'spatial' or 'random'
        self.n_random_individuals = experiment_constants['n_random_individuals_per_generation']
        self.homeostasis_steps = experiment_constants['homeostasis_steps'] if 'homeostasis_steps' in experiment_constants else 1

        self.n_layers = len(self.layers)
        self.base_layer = next((i for i, d in enumerate(self.layers) if d.get('base', False)), None)
        self.population = []
        self.current_generation = 1
        self.num_ids = 0

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.parent_child_distance_history = []
        self.mutation_data_history = []
        self.n_neutral_over_generations = []

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

        unsimulated_state_genotypes, unsimulated_indices = self.get_unsimulated_genotypes()
        init_phenotypes = self.make_seed_phenotypes(unsimulated_state_genotypes.shape[0])

        ##### SIMULATE ON GPUs #####
        print(f'Starting {unsimulated_state_genotypes.shape[0]} simulations...')
        full_phenotypes = simulate(
            unsimulated_state_genotypes, 
            self.n_layers, 
            init_phenotypes)

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        # Evaluate each phenotype's fitness
        phenotypes, fitness_scores = self.evaluate_phenotypes(full_phenotypes)

        # Phenotypes will be (pop_size x self.homeostasis_steps x WORLD_SIZE x WORLD_SIZE)

        parent_child_distances = []
        n_neutral = 0
        n_all_dead_or_all_alive = 0
        # Set the fitness and simulated flag for each of the solutions just evaluated 
        for i, idx in enumerate(unsimulated_indices):
            self.population[idx].set_fitness(fitness_scores[i])
            self.population[idx].set_simulated(True)
            self.population[idx].set_phenotype(phenotypes[i]) # phenotype is now binarized last homeostasis steps of base layer
            # Get actual parent Solution object from population using parent_id
            parent = next((sol for sol in self.population if sol.id == self.population[idx].parent_id), None)
            if parent is not None:
                if (phenotypes[i] == parent.phenotype).all():
                    n_neutral += 1
                parent_child_distances.append(self.population[idx].get_distance_from_parent(parent))

            if fitness_scores[i] == 1000000:
                n_all_dead_or_all_alive += 1
            elif self.homeostasis_steps > 1:
                # Check if the phenotype is the same for all homeostasis steps
                if all([(phenotypes[i][j] == phenotypes[i][0]).all() for j in range(self.homeostasis_steps)]):
                    self.population[idx].set_homeostatic(True)
                    self.population[idx].set_fitness(self.population[idx].fitness - 10000)

        mean_fitness = np.mean([sol.fitness for sol in self.population])
        print('Average fitness:', mean_fitness)
        print('Min fitness: ', min([sol.fitness for sol in self.population]))
        print('Average age:', np.mean([sol.age for sol in self.population]))
        print('Number neutral: ', n_neutral, '/', self.target_population_size, ' (', n_neutral / self.target_population_size, ')')
        print('Number homeostatic: ', np.sum([sol.homeostatic for sol in self.population]), '/', self.target_population_size, ' (', np.sum([sol.homeostatic for sol in self.population]) / self.target_population_size, ')')
        print('Number all dead or all alive: ', n_all_dead_or_all_alive, '/', self.target_population_size, ' (', n_all_dead_or_all_alive / self.target_population_size, ')')
        # Reduce the population
        self.reduce_population()
        # Increment ages by 1
        for sol in self.population:
            sol.increment_age()
        # Extend the population using tournament selection
        aggregate_mutation_data = self.extend_population()

        self.best_fitness_history.append(self.best_solution())
        self.mean_fitness_history.append(mean_fitness)
        self.parent_child_distance_history.append(parent_child_distances)
        self.mutation_data_history.append(aggregate_mutation_data)
        self.n_neutral_over_generations.append(n_neutral)


    def initialize_population(self):
        # Initialize target_population_size random solutions
        self.population = [
            Solution(layers=self.layers, id=self.get_available_id()) for _ in range(self.target_population_size)
        ]

    def generate_new_individuals(self):
        new_individuals = []
        mutation_data = []
        # 1 - Breed: do tournament selection
        for _ in range(self.target_population_size):
            # Randomly select an individual using tournament selection
            parent = self.tournament_select()
            child = parent.make_offspring(self.get_available_id())
            mutation_data.append(child.mutation_info)
            new_individuals.append(child)

        return new_individuals, mutation_data

    def extend_population(self):
        new_individuals, mutation_data = self.generate_new_individuals()

        self.population += new_individuals

        # Add N random new individuals
        self.population += [Solution(layers=self.layers, id=self.get_available_id()) for _ in range(self.n_random_individuals + 1)]
        
        aggregate_mutation_data = {
            'layer': dict(Counter([mutation_info['layer'] for mutation_info in mutation_data])),
            'kind': dict(Counter([mutation_info['kind'] for mutation_info in mutation_data])),
        }
        return aggregate_mutation_data


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
        unsimulated_state_genotypes = [
            sol.state_genotype for sol in self.population if not sol.been_simulated
        ]
        unsimulated_indices = [
            i for i, sol in enumerate(self.population) if not sol.been_simulated
        ]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(unsimulated_state_genotypes, dtype=np.float32), unsimulated_indices

    def get_target_shape(self):
        """
        Returns the target shape using self.shape and the resolution of self.base_layer.
        Always returns 64x64 numpy array. 
        """

        resolution = self.layers[self.base_layer]['res']
        target_size = WORLD_SIZE // resolution

        assert resolution in [2**n for n in range(int(np.log2(WORLD_SIZE)) + 1)]

        if self.shape == 'square':
            target = create_square(target_size)
        elif self.shape == 'diamond':
            target = create_diamond(target_size)
        elif self.shape == 'circle':
            target = create_hollow_circle(target_size)
        elif self.shape == 'plus':
            target = create_plus(target_size)
        elif self.shape == 'complex':
            target = create_complex(target_size)

        # Upsize back to WORLD_SIZE x WORLD_SIZE because that's how we're comparing 
        while target.shape[0] < WORLD_SIZE: 
            target = np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)

        # Repeat the target for the number of homeostasis steps
        target = np.stack([target] * self.homeostasis_steps)
        
        assert (target.shape) == (self.homeostasis_steps, WORLD_SIZE, WORLD_SIZE)

        return target
        

    def evaluate_phenotypes(self, full_phenotypes):
        """Score a set of phenotypes generated by the simulate function."""
        target = self.get_target_shape()

        # Infer pop_size from phenotypes
        pop_size = full_phenotypes.shape[0]
        # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
        assert full_phenotypes.shape == (pop_size, NUM_STEPS, self.n_layers, WORLD_SIZE, WORLD_SIZE)

        # Allocate space for results.
        fitness_scores = np.zeros(pop_size, dtype=np.uint32)

        # Get the final homeostasis states of the base layer for each individual
        phenotypes = full_phenotypes[:, -self.homeostasis_steps:, self.base_layer] > 0

        # For each individual in the population...
        for i in range(pop_size):
            # Look at just the final state of the layer0 part of the phenotype.
            # Compare it to the target image, and sum up the deltas to get the
            # final fitness score (lower is better, 0 is a perfect score).
            fitness_scores[i] = np.sum(np.abs(target - phenotypes[i]))

            # Select against all-dead or all-alive phenotypes
            if (phenotypes[i] == 0).all() or (phenotypes[i] == 1).all() or fitness_scores[i] == 10240:
                fitness_scores[i] = 1000000

        return phenotypes, fitness_scores


    # @functools.cache
    def make_seed_phenotypes(self, n):
        """Starting phenotypes to use by default (one ALIVE cell in middle)."""
        # For each inidividual, capture phenotype development over NUM_STEPS. Each
        # phenotype has NUM_LAYERS layers which are all WORLD_SIZE x WORLD_SIZE
        # squares. Layer0 is the "ground truth" of the CA while layers 1 and 2
        # represent a sort of hierarchical internal state for the organism. Layers
        # 1 and 2 are conceptually smaller than layer0 (1/4 and 1/8 respectively),
        # but are represented using arrays of the same size for simplicity.
        phenotypes = np.full(
            (n, NUM_STEPS, self.n_layers, WORLD_SIZE, WORLD_SIZE),
            DEAD, dtype=np.float32)
        
        middle_start = WORLD_SIZE // 2
        middle_end = middle_start + self.layers[self.base_layer]['res']

        # Use a single ALIVE pixel in the middle of the CA world as the initial
        # phenotype state for all individuals in the population.
        for i in range(n):
            phenotypes[i][0][self.base_layer][middle_start:middle_end, middle_start:middle_end] = ALIVE

        return phenotypes

    def pickle_afpo(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def get_available_id(self):
        self.num_ids += 1
        return self.num_ids

    def best_solution(self):
        return min([sol for sol in self.population if sol.fitness is not None])
