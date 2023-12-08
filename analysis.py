import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams.update({'font.size': 14})



def extract_fitness_values(file_path):
    fitness_values = []
    with open(file_path, 'rb') as pf:
        afpo = pickle.load(pf)
        fitness_history = afpo.best_fitness_history
        for solution_object in fitness_history:
            fitness_values.append(solution_object.fitness)
    return fitness_values


def plot_average_fitness(folders, label_names):
    plt.rcParams.update({'font.size': 20})  # Set font size

    for i, folder_name in enumerate(folders):                                               
        all_fitness_values = []
        file_list = [f for f in os.listdir(folder_name) if f.endswith('.pkl')]
        for file_name in file_list:                                   
            file_path = os.path.join(folder_name, file_name)
            fitness_values = extract_fitness_values(file_path)
            all_fitness_values.append(fitness_values)

        fitness_groups = list(zip(*all_fitness_values))                                                             
        average_fitnesses = [sum(fitness_group) / len(fitness_group) for fitness_group in fitness_groups]           
        plt.plot(average_fitnesses, label=label_names[i])

        confidence_interval = 1.96 * np.std(all_fitness_values, axis=0) / np.sqrt(len(all_fitness_values))
        generations = range(len(average_fitnesses))
        plt.fill_between(generations, average_fitnesses - confidence_interval, average_fitnesses + confidence_interval, alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel('Average Loss')
    plt.ylim(250, 1050)
    plt.legend()
    plt.gcf().set_size_inches(10, 8)  # Increase plot size
    plt.savefig(f'fitness_curve_{label_names[0]}_vs_{label_names[1]}.png')
    plt.show()


def strip_plot(folders, label_names):
    plt.rcParams.update({'font.size': 20})  # Set font size

    np.random.seed(0)
    all_max_fit_values = []
    for i, folder_name in enumerate(folders):
        max_fit_list = []
        file_list = [f for f in os.listdir(folder_name) if f.endswith('.pkl')]
        for file_name in file_list:
            file_path = os.path.join(folder_name, file_name)
            fitness_values = extract_fitness_values(file_path)
            max_fit_list.append(fitness_values[-1])
        all_max_fit_values.append(max_fit_list)

    ax = sns.stripplot(data=all_max_fit_values, jitter=True, size=15, alpha=0.7)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names)
    plt.ylabel('Minimum Loss')
    plt.ylim(250, 1050)
    plt.gcf().set_size_inches(10, 8)  # Increase plot size
    plt.savefig(f'strip_plot_{label_names[0]}_vs_{label_names[1]}.png')
    plt.show()


def box_plot(folders, label_names):
    plt.rcParams.update({'font.size': 20})  # Set font size

    all_max_fit_values = []
    for folder_name in folders:
        max_fit_list = []
        file_list = [f for f in os.listdir(folder_name) if f.endswith('.pkl')]
        for file_name in file_list:
            file_path = os.path.join(folder_name, file_name)
            fitness_values = extract_fitness_values(file_path)
            max_fit_list.append(fitness_values[-1])
        all_max_fit_values.append(max_fit_list)
    
    plt.boxplot(all_max_fit_values, labels=label_names)
    plt.ylabel('Minimum Loss')
    plt.ylim(250, 1050)
    plt.gcf().set_size_inches(10, 8)  # Increase plot size
    plt.savefig(f'box_plot_{label_names[0]}_vs_{label_names[1]}.png')
    plt.show()



# Define datasets
layers_3_growth_true = 'experiments/growth-exp/Control'
layers_3_growth_false = 'experiments/growth-exp/Experiment'

layers_2_growth_true = 'experiments/two_layers/Two_Layers'

layers_1_growth_true = 'experiments/one_layer/Experiment'
layers_1_growth_false = 'experiments/one_layer/Control'

# (3 Layers, Growth) vs (2 Layers, Growth) vs (1 Layer, Growth) 
plot_average_fitness([layers_3_growth_true, layers_2_growth_true, layers_1_growth_true], ['3 Layers, Growth', '2 Layers, Growth','1 Layer, Growth'])
box_plot([layers_3_growth_true, layers_2_growth_true, layers_1_growth_true], ['3 Layers \n Growth', '2 Layers \n Growth','1 Layer \n Growth'])
strip_plot([layers_3_growth_true, layers_2_growth_true, layers_1_growth_true], ['3 Layers \n Growth', '2 Layers \n Growth','1 Layer \n Growth'])

# (3 Layers, No Growth) vs (3 Layers, No Growth)
plot_average_fitness([layers_3_growth_false, layers_1_growth_false], ['3 Layers, No Growth', '1 Layer, No Growth'])
box_plot([layers_3_growth_false, layers_1_growth_false], ['3 Layers \n No Growth', '1 Layer \n No Growth'])
strip_plot([layers_3_growth_false, layers_1_growth_false], ['3 Layers \n No Growth', '1 Layer \n No Growth'])


# Compare growth to no growth
plot_average_fitness([layers_3_growth_true, layers_3_growth_false, layers_1_growth_true, layers_1_growth_false], ['3 Layers, Growth', '3 Layer, No Growth', '1 Layer, Growth', '1 Layer, No Growth'])
strip_plot([layers_3_growth_true, layers_3_growth_false, layers_1_growth_true, layers_1_growth_false], ['3 Layers \n Growth', '3 Layer \n No Growth', '1 Layer \n Growth', '1 Layer \n No Growth'])









