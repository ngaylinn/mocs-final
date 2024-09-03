from simulation import simulate
from afpo import Solution
import numpy as np
import tracemalloc
import pickle
import random

class NeutralEngine:
    """
    Takes a genotype and walks randomly, in many directions,
    along the neutral network of the phenotype. 
    - Does not need "fitness"
    """
    def __init__(self, exp, init_solution, init_genotype, mutate_layers=None):
        self.experiment = exp
        self.init_genotype = init_genotype
        self.mutate_layers = mutate_layers

        self.child_population = []
        self.parent_population = []
        self.beneficial_solutions = []
        self.num_ids = 1
        self.pop_size = 400 # self.experiment.target_population_size
        # self.init_solution = Solution(layers=exp.layers, id=0)
        self.init_solution = init_solution
        self.init_solution.fitness_history = []
        self.init_solution.signaling_distance_history = []
        self.init_solution.phenotype_history = []
        # self.init_solution
        self.init_solution.state_genotype = self.init_genotype
        self.simulate_initial_genotype()

    def run(self, n_steps=10):
        self.init_pop()
        tracemalloc.start()
        for step in range(n_steps):
            self.take_one_neutral_step(step)
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            for stat in top_stats[:3]:
                print(stat)

            # if len(self.parent_population) > 300:
            #     break

    def init_pop(self):
        self.parent_population = [self.init_solution]
        for _ in range(self.pop_size):
            sol = self.init_solution.make_offspring(self.get_available_id(), self.mutate_layers)
            # print(np.sum(sol.state_genotype == self.init_solution.state_genotype), '/', np.prod(sol.state_genotype.shape))
            self.child_population.append(sol)

    def take_one_neutral_step(self, step):
        """
        Takes the initial genotype, generates N random children, 
        evaluates them for neutrality, saturates the population
        with self.pop_size single-step-neutral children.
        """
        init_phenotypes = self.experiment.make_seed_phenotypes(self.pop_size)
        state_genotypes = np.array([sol.state_genotype for sol in self.child_population])

        
        # Simulate the random children 
        phenotypes = simulate(
            state_genotypes, 
            self.experiment.n_layers, 
            self.child_population[0].around_start, 
            self.child_population[0].above_start, 
            init_phenotypes, 
            np.array(self.experiment.below_map),
            np.array(self.experiment.above_map)) # ,
            # self.experiment.above_map)
        
        # fitness_scores, binarized_phenotypes = self.experiment.evaluate_phenotypes(phenotypes)
        fitness_scores = self.experiment.evaluate_phenotypes(phenotypes)

        parent_ids = np.unique([sol.parent_id for sol in self.child_population])
        parent_state_genotypes = np.array([sol.state_genotype for sol in self.parent_population if sol.id in parent_ids])
        init_parents_phenotypes = self.experiment.make_seed_phenotypes(len(parent_ids))
        
        parent_phenotypes = simulate(
            parent_state_genotypes, 
            self.experiment.n_layers, 
            self.parent_population[0].around_start, 
            self.parent_population[0].above_start, 
            init_parents_phenotypes, 
            np.array(self.experiment.below_map),
            np.array(self.experiment.above_map)) # ,
            # self.experiment.above_map)

        # Evaluate children
        neutral_signaling_children = []
        for i, child in enumerate(self.child_population):
            # Binarize and evaluate phenotype
            binarized_phenotype = phenotypes[i, -5:, self.experiment.base_layer] > 0
            child.set_phenotype(binarized_phenotype)
            child.set_fitness(fitness_scores[i])

            # self.child_population[i].set_full_phenotype(phenotypes[i])

            same_phenotype_as_parent = (binarized_phenotype == self.init_solution.phenotype).all()

            # Calculate parent signaling distance
            parent_id, parent = next(((i, sol) for i, sol in enumerate(self.parent_population) if sol.id == child.parent_id), None)
            parent_phenotype_idx, parent_id = next(((i, parent_id) for i, parent_id in enumerate(parent_ids) if parent_id == child.parent_id), None)

            # print(parent_id, parent_phenotype_idx)
            

            child.signaling_distance = np.sum(np.abs(parent_phenotypes[parent_phenotype_idx] - phenotypes[i]))
            child.full_signaling_distance = np.sum(np.abs(self.init_solution.full_phenotype - phenotypes[i]))
            
            if child.signaling_distance != 0 and same_phenotype_as_parent:
                print(child.signaling_distance)
                child.neutral_counter += 1
                neutral_signaling_children.append(child)

            if fitness_scores[i] < parent.fitness:
                self.beneficial_solutions.append(child)
                print('Beneficial improvement: ', self.init_solution.fitness, ' - ', fitness_scores[i], ' = ', self.init_solution.fitness - fitness_scores[i])
            
        # Print stats
        print(f'Step {step}: ')
        print(np.sum(fitness_scores <= self.init_solution.fitness), ' neutral mutations')
        print(np.sum(fitness_scores < self.init_solution.fitness), ' beneficial mutations')
        print(len(neutral_signaling_children), ' neutral w/ different signaling')
        print(len(self.beneficial_solutions), ' total beneficial solutions found')

        # if len(self.beneficial_solutions) > 0:
        #     self.pickle_ne('./Apr14_beneficial.pkl')

        # Select for the longest neutral paths
        self.select(neutral_signaling_children)
        
        # Generate new children from the neutral children
        self.generate_new_children()

    def select(self, neutral_signaling_children):
        self.parent_population += neutral_signaling_children
        if len(self.parent_population) < self.pop_size:
            return 
        
        # new_parent_pop = []
        new_parent_pop = list(reversed(sorted([(sol.neutral_counter, sol) for i, sol in enumerate(self.parent_population)])))
        # self.parent_population = random.sample(self.parent_population, self.pop_size)
        # while len(new_parent_pop) < 500:
        #     _, max_path_length_solution = self.longest_neutral_walk_from_original()
        #     new_parent_pop.append(max_path_length_solution)
        self.parent_population = [sol for neutral, sol in new_parent_pop[:self.pop_size]]


    def generate_new_children(self):
        n = len(self.parent_population)
        print(n, ' parents')
        print('Max neutral path length: ', max([p.neutral_counter for p in self.parent_population]))
        self.child_population = []
        for _ in range(self.pop_size):
            rand_parent = np.random.randint(n)
            parent = self.parent_population[rand_parent]
            child = parent.make_offspring(self.get_available_id(), self.mutate_layers)
            self.child_population.append(child)

    def simulate_initial_genotype(self):
        init_phenotypes = self.experiment.make_seed_phenotypes(1)

        
        # Simulate the random children 
        phenotypes = simulate(
            np.array([self.init_genotype]), 
            self.experiment.n_layers, 
            self.init_solution.around_start, 
            self.init_solution.above_start, 
            init_phenotypes, 
            self.experiment.below_map,
            self.experiment.above_map) # ,
            # self.experiment.above_map)


        fitness_scores = self.experiment.evaluate_phenotypes(phenotypes)
        print(f'''
        Initial genotype fitness: {fitness_scores[0]}
        ''')
        self.init_solution.set_fitness(fitness_scores[0])
        self.init_solution.set_phenotype(phenotypes[0, -5:, self.experiment.base_layer] > 0)
        self.init_solution.set_full_phenotype(phenotypes[0])
        self.init_solution.full_signaling_distance = 0

    def longest_signaling_distance_from_original(self):
        return max([(sol.full_signaling_distance, sol) for i, sol in enumerate(self.parent_population)])

    def longest_neutral_walk_from_original(self):
        return max([(sol.neutral_counter, sol) for i, sol in enumerate(self.parent_population)])

    def get_available_id(self):
        self.num_ids += 1
        return self.num_ids
    
    def pickle_ne(self, file_name):
        with open(file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)
