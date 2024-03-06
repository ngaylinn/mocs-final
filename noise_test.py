from afpo import AgeFitnessPareto
from hillclimber import HillClimber

params = {
    'optimizer': 'hillclimber',
    'num_trials': 1,
    'target_population_size': 100,
    'max_generations': 50,
    'state_or_growth': None,
    'mutate_layers': None,
    'noise_rate': 0,
    'noise_intensity': 0,
    'sim_steps': 100,
    'shape': 'square',
    'layers': [
        {'res': 1},
        {'res': 2},
        {'res': 4},
        {'res': 8, 'base': True}
    ],
    'use_growth': True,
    'activation': 'sigmoid'
  }

afpo = HillClimber(params)
noise = afpo.generate_noise()

afpo.evolve()

print(noise.shape)
print(noise.min(), noise.max())