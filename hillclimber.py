import functools
import time
from collections import Counter
import pickle
from collections import Counter

import numpy as np

from afpo import Solution, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, DEAD, ALIVE, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from util import create_hollow_circle, create_square, create_diamond

activation2int = {
    'sigmoid': ACTIVATION_SIGMOID,
    'tanh': ACTIVATION_TANH,
    'relu': ACTIVATION_RELU
}

class HillClimber:
    def __init__(self, experiment_constants):
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.use_growth = experiment_constants['use_growth']
        self.activation = experiment_constants['activation']
        self.shape = experiment_constants['shape']
        self.mutate_layers = experiment_constants['mutate_layers']
        self.state_or_growth = experiment_constants['state_or_growth']
        self.noise_rate = experiment_constants['noise_rate']
        self.noise_intensity = experiment_constants['noise_intensity']

        self.n_layers = len(self.layers)
        self.base_layer = next((i for i, d in enumerate(self.layers) if d.get('base', False)), None)
        self.parent_population = {}
        self.children_population = {}
        self.current_generation = 1
        self.num_ids = 0

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.parent_child_distance_history = []
        self.n_neutral_over_generations = []
        self.mutation_data_over_generations = []

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

        unsimulated_growth_genotypes_children, unsimulated_state_genotypes_children, unsimulated_ids_children = self.get_children_genotypes()
        growth_genotypes_parents, state_genotypes_parents, parent_ids = self.get_parent_genotypes()
        init_phenotypes = self.make_seed_phenotypes(unsimulated_growth_genotypes_children.shape[0])
        noise = self.generate_noise()

        rand_id = list(self.children_population.keys())[0]

        ##### SIMULATE ON GPUs #####
        print(f'Starting {self.target_population_size} simulations...')
        children_phenotypes = simulate(
            unsimulated_growth_genotypes_children, 
            unsimulated_state_genotypes_children, 
            self.n_layers, 
            self.base_layer,  
            self.children_population[rand_id].around_start, 
            self.children_population[rand_id].above_start, 
            self.use_growth, 
            init_phenotypes, 
            activation2int[self.activation],
            noise)
        
        # Resimulate parents with new noise
        # i.e. if you want to stick around in the population, you have to endure the noise
        
        if len(self.parent_population) > 0:
            print(f'Resimulating parents...')
            parent_phenotypes = simulate(
                growth_genotypes_parents, 
                state_genotypes_parents, 
                self.n_layers, 
                self.base_layer, 
                self.parent_population[parent_ids[0]].around_start, 
                self.parent_population[parent_ids[0]].above_start, 
                self.use_growth, 
                init_phenotypes, 
                activation2int[self.activation],
                noise)
            parent_fitness_scores = self.evaluate_phenotypes(parent_phenotypes)
            n_same_phenotype = 0
            for i, id in enumerate(parent_ids):
                if (self.parent_population[id].phenotype == (parent_phenotypes[i][-1][self.base_layer] > 0)).all():
                    n_same_phenotype += 1
                self.parent_population[id].set_fitness(parent_fitness_scores[i])
                self.parent_population[id].set_phenotype(parent_phenotypes[i][-1][self.base_layer] > 0)

            print(len(parent_ids), ' parents resimulated')
            print('num same phenotype: ', n_same_phenotype)

        elapsed = time.perf_counter() - start
        lps = (self.target_population_size*2) / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        children_fitness_scores = self.evaluate_phenotypes(children_phenotypes)
        parent_child_distances = []
        # Set the fitness and simulated flag for each of the just-evaluated solutions
        for i, id in enumerate(unsimulated_ids_children):
            self.children_population[id].set_fitness(children_fitness_scores[i])
            self.children_population[id].set_phenotype(children_phenotypes[i][-1][self.base_layer] > 0) # phenotype is now binarized last step of base layer
            # self.children_population[id].set_simulated(True)
            
            # Get actual parent Solution object from population using parent_id
            parent_id = self.children_population[id].parent_id
            parent = self.parent_population[parent_id] if parent_id is not None else None
            if parent is not None:
                parent_child_distances.append(self.children_population[id].get_distance_from_parent(parent))
        
        # Reduce the population by selecting parent or child to remove
        n_neutral_children = self.select()
        self.n_neutral_over_generations.append(n_neutral_children)
        print('num neutral mutations: ', n_neutral_children)

        print('Average fitness:',
              np.mean([sol.fitness for id, sol in self.parent_population.items()]),
              ', Min fitness: ',
              min([sol.fitness for id, sol in self.parent_population.items()]))
        print('Average age:',
              np.mean([sol.age for id, sol in self.parent_population.items()]))

        # Extend the population by mutating the parents
        mutation_data = self.mutate_population()

        self.mutation_data_over_generations.append(mutation_data)
        self.best_fitness_history.append(self.best_solution())
        self.mean_fitness_history.append(np.mean([sol.fitness for id, sol in self.parent_population.items()]))
        self.parent_child_distance_history.append(parent_child_distances)

    def initialize_population(self):
        # Initialize target_population_size random solutions
        for _ in range(self.target_population_size):
            new_id = self.get_available_id()
            self.children_population[new_id] = Solution(layers=self.layers, id=new_id)

    def select(self):
        """
        Look through children, compare fitness with parents, keep the best of the two.
        """
        n_neutral_children = 0
        for child_id, child in self.children_population.items():
            # Get actual parent Solution object from population using parent_id
            parent_id = child.parent_id
            parent = self.parent_population[parent_id] if parent_id is not None else None
            if parent is not None:
                if (child.phenotype == parent.phenotype).all():
                    n_neutral_children += 1
                # if child.fitness == parent.fitness:
                #     n_neutral_children += 1
                if child.fitness <= parent.fitness:
                    del self.parent_population[parent_id]
                    self.parent_population[child_id] = child
            else:
                self.parent_population[child_id] = child

        return n_neutral_children


    def mutate_population(self):
        """
        Mutate each individual in the population  
        """
        mutation_data = []
        self.children_population = {}
        # Make a new child from every parent
        for id, solution in self.parent_population.items():
            new_id = self.get_available_id()
            child = solution.make_offspring(new_id, mutate_layers=self.mutate_layers, state_or_growth=self.state_or_growth)
            mutation_data.append(child.mutation_info)
            self.children_population[new_id] = child

            # print((solution.growth_genotype == child.growth_genotype).all())
            # print((solution.state_genotype == child.state_genotype).all())
            # print(child.mutation_info)

            # print(solution.state_genotype, child.state_genotype)

        aggregate_mutation_data = {
            'type': dict(Counter([mutation_info['type'] for mutation_info in mutation_data])),
            'kind': dict(Counter([mutation_info['kind'] for mutation_info in mutation_data])),
            'layer': dict(Counter([mutation_info['layer'] for mutation_info in mutation_data])),
        }
        print(aggregate_mutation_data)
        return aggregate_mutation_data

    def tournament_select(self):
        """
        Tournament selection randomly chooses two individuals from the population and
        selects the better (based on a primary objective) of the two for reproduction/mutation
        """
        sol1, sol2 = np.random.choice(self.population, 2, replace=False)
        return min(sol1, sol2)


    def get_children_genotypes(self):
        # Filter out just the genotypes that haven't been simulated yet.
        unsimulated_growth_genotypes = [sol.growth_genotype for _, sol in self.children_population.items()]
        unsimulated_state_genotypes = [sol.state_genotype for _, sol in self.children_population.items()]
        unsimulated_ids = [id for id, sol in self.children_population.items()]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(unsimulated_growth_genotypes, dtype=np.float32), np.array(unsimulated_state_genotypes, dtype=np.float32), unsimulated_ids
    
    def get_parent_genotypes(self):
        # Filter out just the genotypes that haven't been simulated yet.
        parent_growth_genotypes = [sol.growth_genotype for _, sol in self.parent_population.items()]
        parent_state_genotypes = [sol.state_genotype for _, sol in self.parent_population.items()]
        parent_ids = [id for id, sol in self.parent_population.items()]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(parent_growth_genotypes, dtype=np.float32), np.array(parent_state_genotypes, dtype=np.float32), parent_ids

    def generate_noise(self):
        shape = (NUM_STEPS, self.n_layers, WORLD_SIZE, WORLD_SIZE)
        noise_matrix = np.zeros(shape)
        mask = np.random.rand(*shape) < self.noise_rate
        normal_values = np.random.normal(0, self.noise_intensity, np.count_nonzero(mask))
        print(normal_values.shape, np.count_nonzero(mask))
        noise_matrix[mask] = normal_values

        for l in range(1,self.n_layers):
            granularity = 2**l
            noise_matrix[:, l] = np.repeat(np.repeat(noise_matrix[:, l, ::granularity, ::granularity], granularity, axis=1), granularity, axis=2)

        return noise_matrix

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

        # Upsize back to 64x64 because that's how we're comparing 
        while target.shape[0] < 64: 
            target = np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)
        assert target.shape[0] == 64

        return target
        

    def evaluate_phenotypes(self, phenotypes):
        """Score a set of phenotypes generated by the simulate function."""
        target = self.get_target_shape()

        # Infer pop_size from phenotypes
        pop_size = phenotypes.shape[0]
        # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
        assert phenotypes.shape == (pop_size, NUM_STEPS, self.n_layers, WORLD_SIZE, WORLD_SIZE)

        # Allocate space for results.
        fitness_scores = np.zeros(pop_size, dtype=np.uint32)

        # For each individual in the population...
        for i in range(pop_size):
            # Look at just the final state of the layer0 part of the phenotype.
            # Compare it to the target image, and sum up the deltas to get the
            # final fitness score (lower is better, 0 is a perfect score).
            fitness_scores[i] = np.sum(np.abs(target - (phenotypes[i][-1][self.base_layer] > 0)))

        return fitness_scores


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


    def pickle_hc(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def get_available_id(self):
        self.num_ids += 1
        return self.num_ids

    def best_solution(self):
        return min([sol for id, sol in self.parent_population.items() if sol.fitness is not None])
