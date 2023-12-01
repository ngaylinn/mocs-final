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