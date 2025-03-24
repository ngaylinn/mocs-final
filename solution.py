import functools
import numpy as np

from simulation import ACTIVATION_SIGMOID, ACTIVATION_TANH, ACTIVATION_RELU

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
        self.homeostatic = False
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

    def set_homeostatic(self, new_homeostatic):
        self.homeostatic = new_homeostatic

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def track_mutation_info(self, layer, c):
        self.mutation_info = {'kind': c, 'layer': layer, 'new_value': self.state_genotype[layer,c]}
        
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
        return len(self.state_genotype[l])

    def get_distance_from_parent(self, parent):
        assert self.parent_id == parent.id
        if self.parent_id is None:
            return 0
        else:
            return np.count_nonzero(self.phenotype != parent.phenotype)

    def randomize_genome(self):
        """
        Structure of genome:
        - state_genotype: (n_layers, n_layers*9)
        """
        self.state_genotype = np.random.random((self.n_layers, self.n_layers*9)).astype(np.float32) * 2 - 1
        self.state_n_weights = np.count_nonzero(self.state_genotype)
        self.total_weights = self.state_n_weights
