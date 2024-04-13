import functools
import time
import pickle
from collections import Counter
import tracemalloc

import numpy as np

from afpo import Solution, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, DEAD, ALIVE, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from util import create_hollow_circle, create_square, create_diamond, create_plus, create_complex

activation2int = {
    'sigmoid': ACTIVATION_SIGMOID,
    'tanh': ACTIVATION_TANH,
    'relu': ACTIVATION_RELU
}

class HillClimber:
    def __init__(self, experiment_constants, exp_directory='.'):
        self.exp_directory = exp_directory
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.activation = experiment_constants['activation']
        self.shape = experiment_constants['shape']
        self.mutate_layers = experiment_constants['mutate_layers']
        self.neighbor_map_type = experiment_constants['neighbor_map_type']

        self.n_layers = len(self.layers)
        self.base_layer = next((i for i, d in enumerate(self.layers) if d.get('base', False)), None)
        self.parent_population = {}
        self.children_population = {}
        self.current_generation = 1
        self.num_ids = 0

        self.best_fitness_history = []
        self.fitness_history = []
        self.parent_child_fitness_history = []
        self.parent_child_distance_history = []
        self.n_neutral_over_generations = []
        self.mutation_data_over_generations = []
        self.beneficial_mutations_over_generations = []

        self.below_map = self.initialize_below_map()

    def evolve(self):
        self.initialize_population()
        
        tracemalloc.start()
        while self.current_generation <= self.max_generations:
            print(f'Generation {self.current_generation}')
            self.evolve_one_generation()
            self.current_generation += 1

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            for stat in top_stats[:3]:
                print(stat)

            if self.current_generation >= 2000 and self.current_generation % 1000 == 0:
                self.pickle_hc(f'{self.exp_directory}/hc_{self.current_generation}.pkl')

        return self.best_solution()

    def evolve_one_generation(self):
        # Actually run the simulations, and time how long it takes.
        start = time.perf_counter()

        unsimulated_children_genotypes = self.get_children_genotypes()
        init_phenotypes = self.make_seed_phenotypes(unsimulated_children_genotypes.shape[0])

        rand_id = list(self.children_population.keys())[0]

        ##### SIMULATE ON GPUs #####
        print(f'Starting {self.target_population_size} simulations...')
        phenotypes = simulate(
            unsimulated_children_genotypes, 
            self.n_layers, 
            self.children_population[rand_id].around_start, 
            init_phenotypes, 
            self.below_map)

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        fitness_scores, binarized_phenotypes = self.evaluate_phenotypes(phenotypes)

        parent_child_distances = []
        parent_child_fitnesses = []
        # Set the fitness and simulated flag for each of the just-evaluated solutions
        for i, id in enumerate(self.children_population):
            self.children_population[id].set_fitness(fitness_scores[i])
            self.children_population[id].set_simulated(True)
            # self.children_population[id].set_full_phenotype(phenotypes[i])
            self.children_population[id].set_phenotype(phenotypes[i][-1][self.base_layer] > 0) # phenotype is now binarized last step of base layer
            # Get actual parent Solution object from population using parent_id
            parent_id = self.children_population[id].parent_id
            parent = self.parent_population[parent_id] if parent_id is not None else None
            if parent is not None:
                parent_child_fitnesses.append((parent.fitness, self.children_population[id].fitness))
                parent_child_distances.append(self.children_population[id].get_distance_from_parent(parent))

        # Reduce the population by selecting parent or child to remove
        n_neutral_children, beneficial_mutations = self.select()
        # Extend the population using tournament selection
        mutation_data = self.mutate_population()

        mean_fitness = np.mean([sol.fitness for id, sol in self.parent_population.items()])
        self.parent_child_fitness_history.append(parent_child_fitnesses)
        self.n_neutral_over_generations.append(n_neutral_children)
        # self.best_fitness_history.append(self.best_solution())
        self.fitness_history.append([sol.fitness for id, sol in self.parent_population.items()])
        # self.mean_fitness_history.append(np.mean([sol.fitness for id, sol in self.parent_population.items()]))
        self.parent_child_distance_history.append(parent_child_distances)
        self.mutation_data_over_generations.append(mutation_data)
        self.beneficial_mutations_over_generations.append(beneficial_mutations)

        print('Average fitness:', mean_fitness)
        print('Min fitness: ', min([sol.fitness for id, sol in self.parent_population.items()]))
        print('Average age:', np.mean([sol.age for id, sol in self.parent_population.items()]))
        print('Proportion neutral: ', n_neutral_children / self.target_population_size)
        # print('Neutral path Counter: ', Counter(beneficial_mutations))
        # print('Neutrals: ', Counter([child.neutral_counter for child_id, child in self.children_population.items()]))

    def initialize_population(self):
        # Make sure all solutions are "interesting"
        # i.e. all solutions' phenotypes are NOT all zeros or all ones
        while len(self.children_population) < self.target_population_size:
            print('Initial population length: ', len(self.children_population))
            print(Counter([sol.fitness for id, sol in self.children_population.items()]))

            # Initialize target_population_size random solutions
            preliminary_pop = {}
            for _ in range(self.target_population_size):
                new_id = self.get_available_id()
                preliminary_pop[new_id] = Solution(layers=self.layers, id=new_id)

            # simulate the preliminary population
            unsimulated_genotypes = np.array([sol.state_genotype for id, sol in preliminary_pop.items()])
            init_phenotypes = self.make_seed_phenotypes(unsimulated_genotypes.shape[0])
            phenotypes = simulate(
                unsimulated_genotypes, 
                self.n_layers, 
                preliminary_pop[list(preliminary_pop.keys())[0]].around_start, 
                init_phenotypes, 
                self.below_map)
            
            fitness_scores, phenotypes = self.evaluate_phenotypes(phenotypes)
            print(Counter(fitness_scores))

            for i, sol_id in enumerate(preliminary_pop):
                if fitness_scores[i] != 999999:
                    preliminary_pop[sol_id].set_fitness(fitness_scores[i])
                    preliminary_pop[sol_id].set_simulated(True)
                    preliminary_pop[sol_id].set_phenotype(phenotypes[i])
                    self.children_population[sol_id] = preliminary_pop[sol_id]
                    # Stop adding to initial population if we've reached target_population_size
                    if len(self.children_population) > self.target_population_size:
                        break



    def generate_new_individuals(self):
        new_individuals = []
        # 1 - Breed: do tournament selection
        # The minus one is to make room for one random individual at the end.
        for _ in range(self.target_population_size - 1):
            # Randomly select an individual using tournament selection
            parent = self.tournament_select()
            new_individuals.append(parent.make_offspring(self.get_available_id()))

        return new_individuals

    def select(self):
        """
        Look through children, compare fitness with parents, keep the best of the two.
        """
        n_neutral_children = 0
        beneficial_mutations = []
        for child_id, child in self.children_population.items():
            # Get actual parent Solution object from population using parent_id
            parent_id = child.parent_id
            parent = self.parent_population[parent_id] if parent_id is not None else None
            if parent is not None:
                # child.signaling_distance = np.sum(np.abs(child.full_phenotype - parent.full_phenotype))
                child.phenotype_distance = self.children_population[child_id].get_distance_from_parent(parent)
                if child.fitness == parent.fitness:
                    n_neutral_children += 1
                    child.neutral_counter += 1
                    
                if child.fitness < parent.fitness:
                    beneficial_mutations.append(child.neutral_counter)
                    child.neutral_counter = 0

                if child.fitness <= parent.fitness:
                    del self.parent_population[parent_id]
                    self.parent_population[child_id] = child
                else:
                    # self.parent_population[parent_id].signaling_distance = 0
                    self.parent_population[parent_id].phenotype_distance = 0
            else:
                self.parent_population[child_id] = child

        for _, parent in self.parent_population.items():
            parent.fitness_history.append(parent.fitness)
            # parent.phenotype_history.append(parent.phenotype)
            # parent.signaling_distance_history.append(parent.signaling_distance)
            parent.phenotype_distance_history.append(parent.phenotype_distance)

        return n_neutral_children, beneficial_mutations


    def mutate_population(self):
        """
        Mutate each individual in the population  
        """
        self.children_population = {}
        # Make a new child from every parent
        for id, solution in self.parent_population.items():
            new_id = self.get_available_id()
            child = solution.make_offspring(new_id=new_id, mutate_layers=self.mutate_layers)
            self.children_population[new_id] = child

        aggregate_mutation_data = {
            'layer': list([solution.mutation_info['layer'] for i, solution in self.children_population.items()]),
            'kind': list([solution.mutation_info['kind'] for i, solution in self.children_population.items()])
        }
        return aggregate_mutation_data


    def get_children_genotypes(self):
        children_genotypes = [sol.state_genotype for _, sol in self.children_population.items()]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(children_genotypes, dtype=np.float32)

    def get_target_shape(self, n):
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

        # Upsize back to 64x64 because that's how we're comparing 
        while target.shape[0] < 64: 
            target = np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)
        assert target.shape[0] == 64

        target = np.array([target for _ in range(n)])

        return target
        

    def evaluate_phenotypes(self, phenotypes, n_timesteps=5):
        """Score a set of phenotypes generated by the simulate function."""
        target = self.get_target_shape(n_timesteps)

        # Infer pop_size from phenotypes
        pop_size = phenotypes.shape[0]
        # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
        assert phenotypes.shape == (pop_size, NUM_STEPS, self.n_layers, WORLD_SIZE, WORLD_SIZE)

        # Allocate space for results.
        fitness_scores = np.zeros(pop_size, dtype=np.uint32)

        binarized_phenotypes = []

        # For each individual in the population...
        # For each individual in the population...
        for i in range(pop_size):
            # Look at just the final state of the layer0 part of the phenotype.
            # Compare it to the target image, and sum up the deltas to get the
            # final fitness score (lower is better, 0 is a perfect score).
            binarized_phenotype = (phenotypes[i, -n_timesteps:, self.base_layer] > 0)
            binarized_phenotypes.append(binarized_phenotype)

            fitness_scores[i] = np.sum(np.abs(target - binarized_phenotype))
            if binarized_phenotype.all(): # All ones
                fitness_scores[i] = 999999
            elif (binarized_phenotype == False).all(): 
                fitness_scores[i] = 999999

        return fitness_scores, np.array(binarized_phenotypes)


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

        # Use a single ALIVE pixel in the middle of the CA world as the initial
        # phenotype state for all individuals in the population.
        for i in range(n):
            phenotypes[i][0][0][WORLD_SIZE // 2, WORLD_SIZE // 2] = ALIVE

        return phenotypes
    
    def initialize_below_map(self):
        below_map = np.zeros((self.n_layers, 4, 3)).astype(int)
        if self.neighbor_map_type == 'random':
            for l in range(self.n_layers):
                for i in range(4):
                    rand_l = np.random.randint(self.n_layers)
                    rand_r_offset = np.random.randint(WORLD_SIZE)
                    rand_c_offset = np.random.randint(WORLD_SIZE)
                    below_map[l, i] = [rand_l, rand_r_offset, rand_c_offset]
        else:
            for l in range(self.n_layers):
                below_map[l, 0] = [l-1, 0, 0]
                below_map[l, 1] = [l-1, 0, 1]
                below_map[l, 2] = [l-1, 1, 0]
                below_map[l, 3] = [l-1, 1, 1]

        return below_map
    
    def pickle_hc(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def get_available_id(self):
        self.num_ids += 1
        return self.num_ids

    def best_solution(self):
        return min([sol for id, sol in self.parent_population.items()])
