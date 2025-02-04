import functools
import numpy as np

activation2int = {
    'sigmoid': ACTIVATION_SIGMOID,
    'tanh': ACTIVATION_TANH,
    'relu': ACTIVATION_RELU
}

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
        self.full_phenotype = None
        self.mutation_info = None
        self.full_signaling_distance = None
        self.signaling_distance = None
        self.phenotype_distance = None
        self.genotype_distance = None
        self.neutral_counter = 0
        self.fitness_history = []
        self.phenotype_history = []
        self.signaling_distance_history = []
        self.phenotype_distance_history = []
        self.neutral_genotype_history = []
        self.randomize_genome()

    def make_offspring(self, new_id, mutate_layers=None, mutate_param=None):
        child = Solution(layers=self.layers, id=new_id, parent_id=self.id)
        child.state_genotype = self.state_genotype.copy()
        if mutate_layers is None:
            if mutate_param is None:
                child.mutate()
            else:
                child.mutate_param(mutate_param)
        else:
            assert [l in range(self.n_layers) for l in mutate_layers]
            child.mutate_layers(mutate_layers)
        child.neutral_counter = self.neutral_counter
        child.fitness_history = self.fitness_history
        child.signaling_distance_history = self.signaling_distance_history
        child.phenotype_history = self.phenotype_history
            
        return child

    def increment_age(self):
        self.age += 1

    def set_phenotype(self, phenotype):
        self.phenotype = phenotype

    def set_full_phenotype(self, full_phenotype):
        self.full_phenotype = full_phenotype

    def set_simulated(self, new_simulated):
        self.been_simulated = new_simulated

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def track_mutation_info(self, layer, c):
        below_range, around_range, above_range = self.get_layer_state_indices(layer)
        if c in range(*below_range):
            kind = 'below'
        elif c in range(*around_range):
            kind = 'around'
        elif c in range(*above_range):
            kind = 'above'
        else:
            kind = None
        self.mutation_info = {'kind': kind, 'layer': layer, 'new_value': self.state_genotype[layer,c]}
        
    
    def mutate_layers(self, layers):
        """
        Mutate a specific layer's genome by one weight
        """
        layer = np.random.choice(layers)
        random_nonzero_indices = np.transpose(np.nonzero(self.state_genotype[layer]))
        c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))][0]
        self.state_genotype[layer, c] = np.random.random() * 2 - 1
        self.track_mutation_info(layer, c)
        
    def mutate_param(self, param_number):
        '''
        param_number - specifies which parameter in the genome
        to mutate if the parameters were flattened into a 1d array
        '''
        # Get non-zero elements of genotype encoding
        nonzero_indices = np.transpose(np.nonzero(self.state_genotype))
        genotype_size = len(nonzero_indices)

        # Make sure the "parameter number" 
        assert param_number in range(genotype_size)

        # Mutate that parameter randomly...
        layer, c = nonzero_indices[param_number]
        self.state_genotype[layer, c] = np.random.random() * 2 - 1
        self.track_mutation_info(layer, c)

    def mutate(self):
        random_nonzero_indices = np.transpose(np.nonzero(self.state_genotype))
        layer, c = random_nonzero_indices[np.random.choice(len(random_nonzero_indices))]
        self.state_genotype[layer, c] = np.random.random() * 2 - 1
        self.track_mutation_info(layer, c)

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
        n_below = 4 # 0 if l == 0 else int((self.layers[l]['res'] / self.layers[l-1]['res'])**2)
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
        """
        max_below = max([self.get_layer_n_params(l)[0] for l in range(self.n_layers)])
        max_around = max([self.get_layer_n_params(l)[1] for l in range(self.n_layers)])
        max_above = max([self.get_layer_n_params(l)[2] for l in range(self.n_layers)])

        # Starting indices for neighborhood and above parameters
        self.around_start = int(max_below)
        self.above_start = int(max_below + max_around)

        total_param_space = int(max_above + max_around + max_below)
        self.state_genotype = np.random.random((self.n_layers, total_param_space)).astype(np.float32) * 2 - 1

        # Mask state genome
        self.state_genotype[0, 0:self.around_start] = 0
        self.state_genotype[self.n_layers-1, self.above_start:] = 0

        self.state_n_weights = np.count_nonzero(self.state_genotype)
        self.total_weights = self.state_n_weights

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