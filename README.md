# Hierarchical Neural CA Model of Morphogenesis

## Specifying an Experiment

Make an experiment file structured like this:
```
{
    'Control':
        {
            'num_trials': 5,
            'target_population_size': 500,
            'max_generations': 2000,
            'layers': 1,
            'use_growth': True
        },
    'Experiment':
        {
            'num_trials': 5,
            'target_population_size': 500,
            'max_generations': 2000,
            'layers': 3,
            'use_growth': True
        }
}
```

The above file specifies an experiment with two experiment arms: the Control, and the Experiment. The two only differ in the number of layers the HNCA use to compare the evolutionary performance of a single-layer NCA and a 3-layer HNCA. 

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

