# Hierarchical Neural CA Model of Morphogenesis

GECCO 2024: https://dl.acm.org/doi/pdf/10.1145/3638529.3654150

Reach out to corresponding author at kamtb28@gmail.com

## Specifying an Experiment

Make an experiment file structured like this:
```
{
  'Experiment': {
    'optimizer': 'afpo',
    'num_trials': 20,
    'target_population_size': 500,
    'max_generations': 500,
    'n_random_individuals_per_generation': 100,
    'mutate_layers': None,
    'neighbor_map_type': 'spatial',
    'sim_steps': 100,
    'shape': 'circle',
    'layers': [
        {'res': 1, 'base': True},
        {'res': 2},
        {'res': 4}, 
        {'res': 8}
    ],
    'activation': 'sigmoid'
  },
  'Control': {
    'optimizer': 'afpo',
    'num_trials': 20,
    'target_population_size': 500,
    'max_generations': 500,
    'n_random_individuals_per_generation': 100,
    'mutate_layers': None,
    'neighbor_map_type': 'spatial',
    'sim_steps': 100,
    'shape': 'circle',
    'layers': [
        {'res': 1, 'base': True},
    ],
    'activation': 'sigmoid'
  },
}
```

The above file specifies an experiment with two experiment arms: the Control, and the Experiment. 

'optimizer' takes values either 'afpo' or 'hillclimber'. 
'num_trials' describes how many trials to run, if in parallel (probably keep to 1)
'target_population_size' determines population size (or, in the case of hillclimbers, how many hillclimbers to run)
'max_generations' is the number of generations to evolve
'n_random_individuals_per_generation' is used for AFPO, determines how many random individuals to generate per generation
'mutate_layers' takes values either 'spatial' or 'random' 
'neighbor_map_type' takes values either 'spatial' or 'random' -- 'spatial' is a hierarchical structure, 'random' randomly rewires up/down connections
'sim_steps' determines the number of timesteps of a NCA simulation
'shape' determines what shape to evolve to match to
'layers': [ # specification of layers. 'res' is the resolution, WORLD_SIZE is divided by it to get the grid (WORLD_SIZE is defined as a global param in simulation.py)
    {'res': 1, 'base': True}, 
    {'res': 2},
    {'res': 4}, 
    {'res': 8}
],
'activation': 'sigmoid' or 'tanh' or 'relu'




## Running the Experiment Locally

To run the experiment locally, 

```
python3 run_exp.py <experiment_file> <experiment_name>
```

This will create an `experiments` directory, a subdirectory for the experiment, and a subsubdirectory for each of the experiment arms. This is where the experiment results will be placed. 

## Running on the VACC

To run the experiment on the VACC, 

```
sbatch vacc_submit_exp.sh <experiment_file> <experiment_name>
```

**This will spawn a single VACC job for every trial specified in the experiment.** In the example above, 10 VACC jobs will be spawned (5 trials for each experiment arm). Be wary of how many resources you're using.

