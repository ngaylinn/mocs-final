import os
import pickle
import matplotlib.pyplot as plt



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
        fitness_groups = list(zip(*all_fitness_values))                                                     # List of tuples where each tuple contains fitness values of its respective generation    
        average_fitness = [sum(fitnesses) / len(fitness_groups) for fitnesses in fitness_groups]            # Calculate the average fitness for each generation
        label_name = folder_name
        plt.plot(average_fitness, label=label_name)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.title('Fitness Curve Comparison between Contrlo and Experiment')
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
    plt.boxplot(all_max_fit_values, labels=folders)
    plt.ylabel('Maximum Fitness')
    plt.title('Maximum Fitness Comparison between Control and Experiment')
    plt.show()



# Example usage of fitness_curve function:
folders = ['Control', 'Experiment']
plot_average_fitness(folders)

# Example usage of box_lot
box_plot(folders)