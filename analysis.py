import os
import pickle
import matplotlib.pyplot as plt
import numpy as np



def extract_fitness_values(file_path):
    fitness_values = []
    with open(file_path, 'rb') as pf:
        afpo = pickle.load(pf)
        fitness_history = afpo.best_fitness_history
        for solution_object in fitness_history:
            fitness_values.append(solution_object.fitness)
    return fitness_values


def plot_average_fitness(folders):
    for folder_name in folders:
        all_fitness_values = []
        file_list = [f for f in os.listdir(folder_name) if f.endswith('.pkl')]
        for file_name in file_list:
            file_path = os.path.join(folder_name, file_name)
            fitness_values = extract_fitness_values(file_path)
            all_fitness_values.append(fitness_values)

        # List of tuples where each tuple contains fitness values of its respective generation. Calculate average fitness of each tuple
        fitness_groups = list(zip(*all_fitness_values))                                                             
        average_fitnesses = [sum(fitness_group) / len(fitness_group) for fitness_group in fitness_groups]           
        label_name = folder_name.split('/', 2)[2]
        plt.plot(average_fitnesses, label=label_name)

        # Fill 95% confidence interval
        confidence_interval = 1.96 * np.std(all_fitness_values, axis=0) / np.sqrt(len(all_fitness_values))
        generations = range(len(average_fitnesses))
        plt.fill_between(generations, average_fitnesses - confidence_interval, average_fitnesses + confidence_interval, alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.title('Fitness Curve Comparison between Control and Experiment')
    plt.show()


def box_plot(folders):
    all_max_fit_values = []
    for folder_name in folders:
        max_fit_list = []
        file_list = [f for f in os.listdir(folder_name) if f.endswith('.pkl')]
        for file_name in file_list:
            file_path = os.path.join(folder_name, file_name)
            fitness_values = extract_fitness_values(file_path)
            max_fit_list.append(fitness_values[-1])
        all_max_fit_values.append(max_fit_list)
    label_names = [folder.split('/', 2)[2] for folder in folders]
    plt.boxplot(all_max_fit_values, labels=label_names)
    plt.ylabel('Maximum Fitness')
    plt.title('Maximum Fitness Comparison between Control and Experiment')
    plt.show()



# Example usage of fitness_curve function:
folders = ['experiments/one_layer/Control', 'experiments/one_layer/Experiment']
plot_average_fitness(folders)

# Example usage of box_lot
box_plot(folders)