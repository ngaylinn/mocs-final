import time
from hillclimber import HillClimber 

params = {
    'target_population_size': 500,
    'max_generations': 2,
    'mutate_layers': [3],
    'layers': [
      {'res': 1},
      {'res': 2},
      {'res': 4},
      {'res': 8, 'base': True},
    ],
    'sim_steps': 100,
    'activation': 'sigmoid',
    'shape': 'circle',
    'neighbor_map_type': 'spatial'
}

N_TOTAL = 5000
total_valid_pc_pairs = 0
total_neutral = 0 

start = time.time()

while total_valid_pc_pairs < N_TOTAL:
    print(f'========={total_valid_pc_pairs}/{N_TOTAL}==========')
    hc = HillClimber(params)
    hc.evolve()
    
    n_neutral = hc.n_neutral_over_generations[-1]
    total_neutral += n_neutral
    total_valid_pc_pairs += params['target_population_size']

neutral_proportion = total_neutral / N_TOTAL
layers_str = params['mutate_layers']
print(f'Neutrality for mutate_layers={layers_str}: {neutral_proportion}')
end = time.time()
print(f'Time: {end-start}')
