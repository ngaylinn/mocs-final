import time
import math

import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
from PIL import Image

# Phenotype development unfolds over NUM_STEPS time steps.
NUM_STEPS = 100
# This code models hierarchical CAs with 1, 2, 3, or 4 layers.
NUM_LAYERS = 5
# Phenotypes are all 64x64 squares.
WORLD_SIZE = 64

# All cells in all layers have one of these two states.
ALIVE = 1
DEAD = 0

# Max number of threads per block on a 8.6cc GPU
MAX_THREADS_PER_BLOCK = 1024
# Break up each row into COL_BATCH_SIZE groups of COLS_PER_THREAD cells each.
COL_BATCH_SIZE = MAX_THREADS_PER_BLOCK // WORLD_SIZE
COLS_PER_THREAD = WORLD_SIZE // COL_BATCH_SIZE

# Each cell can have neighbors on the layer below, the same layer, or the layer
# above, and each of those neighbors gets a single weight value.
NUM_DOWN_WEIGHTS = 4
NUM_AROUND_WEIGHTS = 9
NUM_UP_WEIGHTS = 1
NUM_INPUT_NEURONS = NUM_DOWN_WEIGHTS + NUM_AROUND_WEIGHTS + NUM_UP_WEIGHTS
# Each cell at the bottom level has 5 output neurons
# 4 binary outputs for spreading up/left/right/down, 1 for self signal state
NUM_OUTPUT_NEURONS = 5

# The genome consists of weights for all layers and an activation function
# which is represented with four scalar values.
LAYER_GENOME_SHAPE = (NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS) 

# Genome indices for thersholds of a non-linear activation function:
# LONELINESS_THRESHOLD = 0
# SPAWN_THRESHOLD_LOW = LONELINESS_THRESHOLD + 1
# SPAWN_THRESHOLD_HIGH = SPAWN_THRESHOLD_LOW + 1
# OVERPOPULATION_THRESHOLD = SPAWN_THRESHOLD_HIGH + 1
# Genome indices for neighbor weights:
DOWN_WEIGHTS_START = 0
AROUND_WEIGHTS_START = DOWN_WEIGHTS_START + NUM_DOWN_WEIGHTS
UP_WEIGHTS_START = AROUND_WEIGHTS_START + NUM_AROUND_WEIGHTS

LEFT_SPREAD_WEIGHTS_COL_IDX = 1
RIGHT_SPREAD_WEIGHTS_COL_IDX = 2
UP_SPREAD_WEIGHTS_COL_IDX = 3
DOWN_SPREAD_WEIGHTS_COL_IDX = 4

ACTIVATION_SIGMOID = 0
ACTIVATION_TANH = 1
ACTIVATION_RELU = 2


@cuda.jit
def activate_sigmoid(weighted_sum):
    """If the weighted sum is negative, """
    if weighted_sum <= 0:
        return 0
    return 1 / (1 + np.e ** (-weighted_sum))

@cuda.jit
def activate_tanh(weighted_sum):
    return 0.5 * (math.tanh(weighted_sum) + 1)

@cuda.jit
def activate_relu(weighted_sum):
    return max(0.0, min(1.0, weighted_sum))


@cuda.jit
def look_down(layer, phenotypes, genotypes, growth, down_weights_start, pop_idx, step, row, col, below_map):
    """Compute the weighted sum of this cell's neighbors in the layer below."""
    if layer == 0 and growth == 0:
        return 0
    
    # row, col are in the granularity of the layer

    # The "granularity" of this layer. Layer0 == 1, layer1 == 2, layer2 == 4, layer3 == 8
    g = 1 << layer

    weight_index = down_weights_start
    result = 0

    for i in range(4):
        # Look at the cell that occupies "the same position" as this cell but
        # one layer down.
        new_layer = below_map[layer][i][0]
        new_layer_g = 1 << new_layer
        new_row = ((below_map[layer][i][1] + ((row // g)*g)) % WORLD_SIZE) // new_layer_g
        new_col = ((below_map[layer][i][2] + ((col // g)*g)) % WORLD_SIZE) // new_layer_g
        # new_row = (row + below_map[layer][i][1]) % WORLD_SIZE
        # new_col = (col + below_map[layer][i][2]) % WORLD_SIZE
        neighbor_state = phenotypes[pop_idx][step-1][new_layer][new_row][new_col]
        weight = genotypes[pop_idx, layer][weight_index]
        result += neighbor_state * weight
        weight_index += 1

    # Look at all the cells that occupy "the same position" as this cell but
    # one layer down.
    # for r in range((row // g)*g, (row // g)*g + g):
    #     for c in range((col // g)*g, (col // g)*g + g):
    #         # Do wrap-around bounds checking. This may be inefficient, since
    #         # we're doing extra modulus operations and working on
    #         # non-contiguous memory may prevent coallesced reads. However, it's
    #         # simple, and avoids complications when working with different
    #         # granularities.
    #         r = r % WORLD_SIZE
    #         c = c % WORLD_SIZE
    #         neighbor_state = phenotypes[pop_idx][step-1][layer-1][r][c]
    #         weight = genotypes[pop_idx, layer][weight_index]
    #         result += neighbor_state * weight
    #         weight_index += 1
    return result


@cuda.jit
def look_around(layer, phenotypes, genotypes, around_weights_start, pop_idx, step, row, col):
    """Compute the weighted sum of this cell's neighbors in this layer."""
    # The "granularity" of this layer. Layer0 == 1, layer1 == 2, layer2 == 4.
    g = 1 << layer

    weight_index = around_weights_start
    result = 0
    # Look at a Moore neighborhood around (row, col) in the current layer.
    for r in range(row-g, row+g+1, g):
        for c in range(col-g, col+g+1, g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            neighbor_state = phenotypes[pop_idx][step-1][layer][r][c]
            weight = genotypes[pop_idx, layer][weight_index]
            result += neighbor_state * weight
            weight_index += 1
    return result


@cuda.jit
def look_up(num_layers, layer, phenotypes, genotypes, growth, up_weights_start, pop_idx, step, row, col, above_map):
    """Compute the weighted sum of this cell's neighbors in the layer above."""
    if layer == num_layers - 1 and growth == 0:
        return 0
    
    # The "granularity" of this layer. Layer0 == 1, layer1 == 2, layer2 == 4.
    g = 1 << layer
    
    new_layer = above_map[layer][0]
    new_layer_g = 1 << new_layer
    new_row = ((above_map[layer][1] + ((row // g)*g)) % WORLD_SIZE) // new_layer_g
    new_col = ((above_map[layer][2] + ((col // g)*g)) % WORLD_SIZE) // new_layer_g

    # Look at just the single neighbor in the next layer up.
    neighbor_state = phenotypes[pop_idx][step-1][new_layer][new_row][new_col]
    weight = genotypes[pop_idx, layer, up_weights_start]
    return neighbor_state * weight

@cuda.jit
def get_spread_update(num_layers, layer, phenotypes, growth_genotypes, around_start, above_start, pop_idx, step, row, col, above_map, below_map):
    up_spread_weighted_sum = look_up(num_layers, layer, phenotypes, growth_genotypes, 1, above_start, pop_idx, step, row, col, above_map) 
    up_spread_weighted_sum += look_around(layer, phenotypes, growth_genotypes, around_start, pop_idx, step, row, col) 
    up_spread_weighted_sum += look_down(layer, phenotypes, growth_genotypes, 1, 0, pop_idx, step, row, col, below_map)
    
    down_spread_weighted_sum = look_up(num_layers, layer, phenotypes, growth_genotypes, 1, above_start, pop_idx, step, row, col, above_map) 
    down_spread_weighted_sum += look_around(layer, phenotypes, growth_genotypes, around_start, pop_idx, step, row, col) 
    down_spread_weighted_sum += look_down(layer, phenotypes, growth_genotypes, 1, 0, pop_idx, step, row, col, below_map)
    
    left_spread_weighted_sum = look_up(num_layers, layer, phenotypes, growth_genotypes, 1, above_start, pop_idx, step, row, col, above_map) 
    left_spread_weighted_sum += look_around(layer, phenotypes, growth_genotypes, around_start, pop_idx, step, row, col) 
    left_spread_weighted_sum += look_down(layer, phenotypes, growth_genotypes, 1, 0, pop_idx, step, row, col, below_map)
    
    right_spread_weighted_sum = look_up(num_layers, layer, phenotypes, growth_genotypes, 1, above_start, pop_idx, step, row, col, above_map) 
    right_spread_weighted_sum += look_around(layer, phenotypes, growth_genotypes, around_start, pop_idx, step, row, col) 
    right_spread_weighted_sum += look_down(layer, phenotypes, growth_genotypes, 1, 0, pop_idx, step, row, col, below_map)

    return (left_spread_weighted_sum > 0,
        right_spread_weighted_sum > 0,
        up_spread_weighted_sum > 0,
        down_spread_weighted_sum > 0)


@cuda.jit
def update_cell(num_layers, layer, use_growth, phenotypes, growth_genotypes, state_genotypes, base_layer, around_start, above_start, pop_idx, step, row, col, activation, above_map, below_map):
    """Compute the next state for a single cell in layer0 from prev states."""

    # Calculate the weighted sum of all neighbors.
    down_signal_sum = look_down(layer, phenotypes, state_genotypes, 0, 0, pop_idx, step, row, col, below_map) # Should return 0 for L=0
    around_signal_sum = look_around(layer, phenotypes, state_genotypes, around_start, pop_idx, step, row, col) 
    up_signal_sum = look_up(num_layers, layer, phenotypes, state_genotypes, 0, above_start, pop_idx, step, row, col, above_map)

    signal_sum = down_signal_sum + around_signal_sum + up_signal_sum

    # Update cells to be alive if on L=0 (only if current cell is actually alive)
    alive = phenotypes[pop_idx][step][layer][row][col] != 0
    if layer == base_layer and alive and use_growth:
        # Spread to nearby cells... is this necessary?
        (left, right, up, down) = get_spread_update(num_layers, base_layer, phenotypes, growth_genotypes, around_start, above_start, pop_idx, step, row, col, above_map, below_map)
                
        if left:
            phenotypes[pop_idx][step][layer][(row % WORLD_SIZE)][((col-1) % WORLD_SIZE)] = phenotypes[pop_idx][step][layer][row][col]
        if right:
            phenotypes[pop_idx][step][layer][(row % WORLD_SIZE)][((col+1) % WORLD_SIZE)] = phenotypes[pop_idx][step][layer][row][col]
        if up:
            phenotypes[pop_idx][step][layer][((row-1) % WORLD_SIZE)][(col % WORLD_SIZE)] = phenotypes[pop_idx][step][layer][row][col]
        if down:
            phenotypes[pop_idx][step][layer][((row+1) % WORLD_SIZE)][(col % WORLD_SIZE)] = phenotypes[pop_idx][step][layer][row][col]

    # Actually update the phenotype state for step on layer1 at (row, col).
    if activation == ACTIVATION_SIGMOID:
        phenotypes[pop_idx][step][layer][row][col] = activate_sigmoid(signal_sum)
    elif activation == ACTIVATION_TANH:
        phenotypes[pop_idx][step][layer][row][col] = activate_tanh(signal_sum)
    elif activation == ACTIVATION_RELU:
        phenotypes[pop_idx][step][layer][row][col] = activate_relu(signal_sum)
        

# Max registers can be tuned per device. 64 is the most my laptop can handle.
@cuda.jit(max_registers=64)
def simulation_kernel(growth_genotypes, state_genotypes, phenotypes, num_layers, base_layer, around_start, above_start, use_growth, activation, above_map, below_map):
    """Compute and record the full development process of a population."""
    # Compute indices for this thread.
    pop_idx = cuda.blockIdx.x
    row = cuda.threadIdx.x
    start_col = cuda.threadIdx.y * COLS_PER_THREAD

    # For each step in the simulation (state at step 0 is pre-populated by the
    # caller)...
    for step in range(1, NUM_STEPS):
        # This thread will compute COLS_PER_THREAD contiguous cells from row,
        # starting at start_col.
        for col in range(start_col, start_col + COLS_PER_THREAD):
            # Update the state in every layer this individual uses.
            for layer in range(0, num_layers):
                update_cell(num_layers, layer, use_growth, phenotypes, growth_genotypes, state_genotypes, base_layer, around_start, above_start, pop_idx, step, row, col, activation, above_map, below_map)
        # Make sure all threads have finished computing this step before going
        # on to the next one.
        cuda.syncthreads()

def get_layer_mask(l):
    """Mask the genome (the NN weights) for a given layer"""
    mask = np.zeros(LAYER_GENOME_SHAPE)
    if l == 0:   # L=0: All inputs except below
        mask[AROUND_WEIGHTS_START:][:] = 1
    elif l == (NUM_LAYERS - 1): # Top layer: All inputs except above
        mask[DOWN_WEIGHTS_START:UP_WEIGHTS_START, 0] = 1
    else: # Middle layers: All inputs, single output
        mask[:, 0] = 1

    return mask


def check_granularity(g, image):
    """Returns True iff image is tiled with g x g squares of a single value."""
    # Scale down by sampling every gth cell in both dimensions.
    scaled_down = image[0::g, 0::g]
    # Scale back up by repeating every cell g times in both dimensions.
    scaled_up = np.repeat(np.repeat(scaled_down, g, 0), g, 1)
    # Check whether the original image matches the resampled version.
    if np.array_equal(image, scaled_up):
        return True
    else:
        print('aint equal: ', image, scaled_up)
        return False
    # return np.array_equal(image, scaled_up)


def simulate(growth_genotypes, state_genotypes, num_layers, base_layer, around_start, above_start, use_growth, phenotypes, activation, below_map, above_map):
    """Simulate genotypes and return phenotype videos."""

    # Infer population size from genotypes
    pop_size = growth_genotypes.shape[0]

    # Each individual has a genotype that consists of an activation function
    # and a set of neighbor weights for each layer in the hierarchical CA.

    # assert growth_genotypes.shape == (pop_size, num_layers, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)
    
    # assert genotypes.dtype == np.uint8

    # Each individual is configured to have 0, 1, or 2 extra layers. This way,
    # we can simulate individuals from control and experiment in the same
    # batch. Individuals with fewer layers should complete faster. That could
    # hurt performance, but shouldn't because each individual is handled by
    # MAX_THREADS_PER_BLOCK threads, which should mean that whole warps fall
    # out of the computation together.
    assert type(num_layers) is int
    assert num_layers in range(1, NUM_LAYERS + 1)

    assert type(use_growth) is bool
    assert type(activation) is int

    assert phenotypes.shape == (
        pop_size, NUM_STEPS, num_layers, WORLD_SIZE, WORLD_SIZE)
    
    assert above_map.shape == (num_layers, 3)
    assert below_map.shape == (num_layers, 4, 3)

    # Copy input data from host memory to device memory.
    d_phenotypes = cuda.to_device(phenotypes)
    d_growth_genotypes = cuda.to_device(growth_genotypes)
    d_state_genotypes = cuda.to_device(state_genotypes)
    d_below_map = cuda.to_device(below_map)
    d_above_map = cuda.to_device(above_map)

    # Actually run the simulation for all individuals in parallel on the GPU.
    simulation_kernel[
        # Each grid contains one block per organism.
        (pop_size,),
        # Organize threads into two dimensions. Each thread computes
        # COLS_PER_THREAD cells in the CA. The X dimension is the row within
        # the CA world to compute, and the Y dimension is multiplied by
        # COLS_PER_THREAD to find the first column to start from.
        (WORLD_SIZE, COL_BATCH_SIZE)
    ](d_growth_genotypes, d_state_genotypes, d_phenotypes, num_layers, base_layer, around_start, above_start, use_growth, activation, d_above_map, d_below_map)

    # Copy output data from device memory to host memory.
    phenotypes = d_phenotypes.copy_to_host()
        
    # Layer1 in all phenotypes from all steps of the simulation has a
    # granularity of 2x2.
    if phenotypes.shape[2] > 1:
        assert all(check_granularity(2, p)
                for p in np.reshape(
                    phenotypes[:, :, 1], (-1, WORLD_SIZE, WORLD_SIZE))) 

    # Layer2 in all phenotypes from all steps of the simulation has a
    # granularity of 4x4.
    if phenotypes.shape[2] > 2:
        assert all(check_granularity(4, p)
                for p in np.reshape(
                    phenotypes[:, :, 2], (-1, WORLD_SIZE, WORLD_SIZE)))

    return phenotypes


def make_seed_phenotypes(pop_size, n_layers=3):
    """Starting phenotypes to use by default (one ALIVE cell in middle)."""
    # For each inidividual, capture phenotype development over NUM_STEPS. Each
    # phenotype has NUM_LAYERS layers which are all WORLD_SIZE x WORLD_SIZE
    # squares. Layer0 is the "ground truth" of the CA while layers 1 and 2
    # represent a sort of hierarchical internal state for the organism. Layers
    # 1 and 2 are conceptually smaller than layer0 (1/4 and 1/8 respectively),
    # but are represented using arrays of the same size for simplicity.
    phenotypes = np.full(
        (pop_size, NUM_STEPS, n_layers, WORLD_SIZE, WORLD_SIZE),
        DEAD, dtype=np.float32)

    # Use a single ALIVE pixel in the middle of the CA world as the initial
    # phenotype state for all individuals in the population.
    for i in range(pop_size):
        middle = WORLD_SIZE // 2
        phenotypes[i][0][0][middle][middle] = ALIVE

    return phenotypes

def make_seed_genotypes(pop_size):
    """Starting genotypes: random initialization"""
    # Randomly initialize the NN weights
    genotypes = np.random.random((pop_size, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)).astype(np.float32) * 2 - 1
    
    # Mask out the weights of layers 1 and 2
    for l in range(NUM_LAYERS):
        genotypes[:, l] = genotypes[:, l] * get_layer_mask(l)

    return genotypes


def compute_fitness(phenotypes, target):
    """Score a set of phenotypes generated by the simulate function."""
    # Infer pop_size from phenotypes
    pop_size = phenotypes.shape[0]
    # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
    assert phenotypes.shape == (
        pop_size, NUM_STEPS, NUM_LAYERS, WORLD_SIZE, WORLD_SIZE)
    assert target.shape == (WORLD_SIZE, WORLD_SIZE)

    # Allocate space for results.
    fitness_scores = np.zeros(pop_size, dtype=np.uint32)

    # For each individual in the population...
    for i in range(pop_size):
        # Look at just the final state of the layer0 part of the phenotype.
        # Compare it to the target image, and sum up the deltas to get the
        # final fitness score (lower is better, 0 is a perfect score).
        fitness_scores[i] = np.sum(np.abs(target - phenotypes[i][-1][0]))

    return fitness_scores


def visualize(phenotype, filename, layer=0):
    def make_frame(frame_data):
        # Scale up the image 4x to make it more visible.
        # frame_data = frame_data.repeat(4, 1).repeat(4, 2)
        layer0, layer1, layer2 = frame_data
        l, w = layer0.shape
        # print(layer0[l//2-1:l//2+5, w//2-1:w//2+5])
        # Make a composite of layers 1 and 2 to overlay on layer0.
        # print(layer1 * 0xffff0000, (layer1 * 0xffff0000).shape)
        # print(layer2 * 0xff00ff00, (layer2 * 0xff00ff00).shape)
        # overlay = np.array(
        #     np.bitwise_or(
        #         layer1 * 0xffff0000,  # layer1 in blue
        #         layer2 * 0xff00ff00), # layer2 in green
        #     dtype=np.uint32)
        # Render layer0 as a black and white image.
        base = np.array(
            np.bitwise_or(
                (layer0 != DEAD) * 0xffffffff,   # DEAD cells are black
                (layer0 == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)
        # Merge the base and overlay images.
        # return layer1
        if layer == 0:
            return Image.fromarray(layer0, mode='RGBA')
        elif layer == 1:
            return Image.fromarray(layer1, mode='RGBA')
        elif layer == 2:
            return Image.fromarray(layer2, mode='RGBA') 
        elif layer == 'base':
            return Image.fromarray(base, mode='RGBA')
        # return Image.blend(
        #     Image.fromarray(base, mode='RGBA'),
        #     Image.fromarray(overlay, mode='RGBA'),
        #     0.3)

    frames = [make_frame(frame_data) for frame_data in phenotype]
    # plt.imshow(frames[11])
    # plt.show()
    # plt.imshow(frames[12])
    # plt.show()
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)
