import functools
import time
import pickle

import numpy as np

from simulation import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from util import create_hollow_circle, create_square, create_diamond

activation2int = {
    'sigmoid': ACTIVATION_SIGMOID,
    'tanh': ACTIVATION_TANH,
    'relu': ACTIVATION_RELU
}

N_RANDOM_INDIVIDUALS = 10

@functools.total_ordering # Sortable by fitness
class Solution:
    def __init__(self, layers, id, parent_id=None):
        self.id = id
        self.parent_id = parent_id
        self.layers = layers
        self.n_layers = len(self.layers)
        self.base_layer = next((i for i, d in enumerate(self.layers) if d.get('base', False)), None)
        self.age = 0
        self.been_simulated = False
        self.fitness = None
        self.phenotype = None
        self.mutation_info = None
        self.randomize_genome()

    def make_offspring(self, new_id, mutate_layers=None, state_or_growth=None):
        child = Solution(layers=self.layers, id=new_id, parent_id=self.id)
        child.state_genotype = self.state_genotype # .copy()
        child.growth_genotype = self.growth_genotype # .copy()
        if mutate_layers is None:
            child.mutate()
        else:
            assert [l in range(self.n_layers) for l in mutate_layers]
            assert state_or_growth is not None
            child.mutate_layers(mutate_layers, state_or_growth=state_or_growth)
            
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
    
    def mutate_layers(self, layers, state_or_growth='state'):
        """
        Mutate a specific layer's genome by one weight
        """
        if state_or_growth == 'state':
            layer = np.random.choice(layers)
            random_nonzero_indices = np.transpose(np.nonzero(self.state_genotype[layer]))
            c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))][0]
            self.state_genotype[layer, c] = np.random.random() * 2 - 1
        else:
            random_nonzero_indices = np.transpose(np.nonzero(self.growth_genotype))
            r, c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))]
            self.growth_genotype[r, c] = np.random.random() * 2 - 1

        L = layer if state_or_growth == 'state' else r
        below_range, around_range, above_range = self.get_layer_state_indices(L)
        if c in range(*below_range):
            kind = 'below'
        elif c in range(*around_range):
            kind = 'around'
        elif c in range(*above_range):
            kind = 'above'
        else:
            kind = None
        self.mutation_info = {'type': state_or_growth, 'kind': kind, 'layer': layer, 'rc': (layer if state_or_growth == 'state' else r, c)}
        

    def mutate(self):
        mutate_growth_weight = np.random.choice([0, 1], p=[self.state_n_weights / self.total_weights, self.growth_n_weights / self.total_weights])

        if mutate_growth_weight:
            random_nonzero_indices = np.transpose(np.nonzero(self.growth_genotype))
            r, c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))]
            self.growth_genotype[r, c] = np.random.random() * 2 - 1
            self.mutation_info = {'type': 'growth', 'layer': self.base_layer, 'kind': None}
        else:
            random_nonzero_indices = np.transpose(np.nonzero(self.state_genotype))
            r, c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))]
            self.state_genotype[r, c] = np.random.random() * 2 - 1
            below_range, around_range, above_range = self.get_layer_state_indices(r)
            if c in range(*below_range):
                kind = 'below'
            elif c in range(*around_range):
                kind = 'around'
            elif c in range(*above_range):
                kind = 'above'
            else:
                kind = None
            self.mutation_info = {'type': 'state', 'layer': r, 'kind': kind}

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

    def get_layer_n_params(self, l):
        n_above = 0 if l == (self.n_layers-1) else 1
        n_below = 0 if l == 0 else (self.layers[l]['res'] / self.layers[l-1]['res'])**2
        n_around = 9 # Moore neighborhood
        return n_below, n_around, n_above

    def get_layer_state_indices(self, l):
        n_above = 0 if l == (self.n_layers-1) else 1
        n_below = 0 if l == 0 else int((self.layers[l]['res'] / self.layers[l-1]['res'])**2)
        n_around = 9 # Moore neighborhood

        # below, around, above
        return ((0,n_below), (n_below, n_below+n_around), (n_below+n_around, n_below+n_around+n_above))

    def get_distance_from_parent(self, parent):
        assert self.parent_id == parent.id
        if self.parent_id is None:
            return 0
        else:
            return np.count_nonzero(self.phenotype != parent.phenotype)

    def randomize_genome(self):
        """
        Structure of genome:
        - state_genotype: (n_layers, max_below + max_around + max_above)
        - growth_genotype: (4, max_below + max_around + max_above)
        """
        max_below = max([self.get_layer_n_params(l)[0] for l in range(self.n_layers)])
        max_around = max([self.get_layer_n_params(l)[1] for l in range(self.n_layers)])
        max_above = max([self.get_layer_n_params(l)[2] for l in range(self.n_layers)])

        # Starting indices for neighborhood and above parameters
        self.around_start = int(max_below)
        self.above_start = int(max_below + max_around)

        total_param_space = int(max_above + max_around + max_below)
        self.state_genotype = np.random.random((self.n_layers, total_param_space)).astype(np.float32) * 2 - 1
        self.growth_genotype = np.random.random((4, total_param_space)).astype(np.float32) * 2 - 1

        # Mask growth genome
        if self.base_layer == 0: # Mask away below if base layer is layer 0
            self.growth_genotype[:, 0:self.around_start] = 0
        if self.base_layer == (self.n_layers - 1):
            self.growth_genotype[:, self.above_start:] = 0

        # Mask state genome
        self.state_genotype[0, 0:self.around_start] = 0
        self.state_genotype[self.n_layers-1, self.above_start:] = 0

        self.state_n_weights = np.count_nonzero(self.state_genotype)
        self.growth_n_weights = np.count_nonzero(self.growth_genotype)
        self.total_weights = self.state_n_weights + self.growth_n_weights


    def randomize_growth_genome(self):
        size = self.get_layer_n_params(self.base_layer)
        self.growth_genotype = np.random.random((size, 4)).astype(np.float32) * 2 - 1 # 4 for up,down,left,right spread

    def randomize_state_genome(self):
        self.state_genotype = np.random.random((self.n_layers, max([self.get_layer_n_params(l) for l in range(self.n_layers)]))).astype(np.float32) * 2 - 1
        self.state_genotype_mask = np.zeros_like(self.state_genotype)
        for l in range(self.n_layers):
            below_index_range, neighborhood_index_range, above_index_range = self.get_layer_state_indices(l)
            print(l, below_index_range, neighborhood_index_range, above_index_range)
            self.state_genotype_mask[l, below_index_range[0]:below_index_range[1]] = 1
            self.state_genotype_mask[l, neighborhood_index_range[0]:neighborhood_index_range[1]] = 1
            self.state_genotype_mask[l, above_index_range[0]:above_index_range[1]] = 1

        self.state_genotype *= self.state_genotype_mask


class AgeFitnessPareto:
    def __init__(self, experiment_constants):
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.use_growth = experiment_constants['use_growth']
        self.activation = experiment_constants['activation']
        self.shape = experiment_constants['shape']
        self.neighbor_map_type = experiment_constants['neighbor_map_type'] # 'spatial' or 'random'

        self.n_layers = len(self.layers)
        self.base_layer = next((i for i, d in enumerate(self.layers) if d.get('base', False)), None)
        self.population = []
        self.current_generation = 1
        self.num_ids = 0

        self.best_fitness_history = []
        self.parent_child_distance_history = []

        self.below_map = self.initialize_below_map()
        self.above_map = self.initialize_above_map()

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

        unsimulated_growth_genotypes, unsimulated_state_genotypes, unsimulated_indices = self.get_unsimulated_genotypes()
        init_phenotypes = self.make_seed_phenotypes(unsimulated_growth_genotypes.shape[0])

        ##### SIMULATE ON GPUs #####
        print(f'Starting {self.target_population_size} simulations...')
        phenotypes = simulate(
            unsimulated_growth_genotypes, 
            unsimulated_state_genotypes, 
            self.n_layers, 
            self.base_layer,  
            self.population[0].around_start, 
            self.population[0].above_start, 
            self.use_growth, 
            init_phenotypes, 
            activation2int[self.activation],
            self.below_map,
            self.above_map)

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        fitness_scores = self.evaluate_phenotypes(phenotypes)
        parent_child_distances = []
        # Set the fitness and simulated flag for each of the just-evaluated solutions
        for i, idx in enumerate(unsimulated_indices):
            self.population[idx].set_fitness(fitness_scores[i])
            self.population[idx].set_simulated(True)
            self.population[idx].set_phenotype(phenotypes[i][-1][self.base_layer] > 0) # phenotype is now binarized last step of base layer
            # Get actual parent Solution object from population using parent_id
            parent = next((sol for sol in self.population if sol.id == self.population[idx].parent_id), None)
            if parent is not None:
                parent_child_distances.append(self.population[idx].get_distance_from_parent(parent))

        print('Average fitness:',
              np.mean([sol.fitness for sol in self.population]),
              ', Min fitness: ',
              min([sol.fitness for sol in self.population]))
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
        self.parent_child_distance_history.append(parent_child_distances)

    def initialize_population(self):
        # Initialize target_population_size random solutions
        self.population = [
            Solution(layers=self.layers, id=self.get_available_id()) for _ in range(self.target_population_size)
        ]

    def generate_new_individuals(self):
        new_individuals = []
        # 1 - Breed: do tournament selection
        # The minus one is to make room for one random individual at the end.
        for _ in range(self.target_population_size - 1):
            # Randomly select an individual using tournament selection
            parent = self.tournament_select()
            new_individuals.append(parent.make_offspring(self.get_available_id()))

        return new_individuals

    def extend_population(self):
        new_individuals = self.generate_new_individuals()

        self.population += new_individuals

        # Add N_RANDOM_INDIVIDUALS new random individuals
        self.population += [Solution(layers=self.layers, id=self.get_available_id()) for _ in range(N_RANDOM_INDIVIDUALS)]


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
        unsimulated_growth_genotypes = [
            sol.growth_genotype for sol in self.population if not sol.been_simulated
        ]
        unsimulated_state_genotypes = [
            sol.state_genotype for sol in self.population if not sol.been_simulated
        ]
        unsimulated_indices = [
            i for i, sol in enumerate(self.population) if not sol.been_simulated
        ]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(unsimulated_growth_genotypes, dtype=np.float32), np.array(unsimulated_state_genotypes, dtype=np.float32), unsimulated_indices

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
    
    def initialize_below_map(self):
        below_map = np.zeros((self.n_layers, 4, 3)).astype(int)
        if self.neighbor_map_type == 'random':
            for l in range(self.n_layers):
                rand_l = np.random.randint(self.n_layers)
                rand_r_offset = np.random.randint(WORLD_SIZE)
                rand_c_offset = np.random.randint(WORLD_SIZE)
                below_map[l, 0] = [rand_l, rand_r_offset, rand_c_offset]
        else:
            for l in range(self.n_layers):
                below_map[l, 0] = [l-1, 0, 0]
                below_map[l, 1] = [l-1, 0, 1]
                below_map[l, 2] = [l-1, 1, 0]
                below_map[l, 3] = [l-1, 1, 1]

        return below_map
    
    def initialize_above_map(self):
        above_map = np.zeros((self.n_layers, 3)).astype(int)
        if self.neighbor_map_type == 'random':
            for l in range(self.n_layers):
                rand_l = np.random.randint(self.n_layers)
                rand_r_offset = np.random.randint(WORLD_SIZE)
                rand_c_offset = np.random.randint(WORLD_SIZE)
                above_map[l] = [rand_l, rand_r_offset, rand_c_offset]
        else:
            for l in range(self.n_layers):
                above_map[l] = [l+1, 0, 0]

        return above_map

    def pickle_afpo(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def get_available_id(self):
        self.num_ids += 1
        return self.num_ids

    def best_solution(self):
        return min([sol for sol in self.population if sol.fitness is not None])
