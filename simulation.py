import time

from numba import cuda
import numpy as np
from PIL import Image

# Phenotype development unfolds over NUM_STEPS time steps.
NUM_STEPS = 100
# This code models hierarchical CAs with depth 1, 2, or 3.
MAX_DEPTH = 3
# Phenotypes are all 64x64 squares.
WORLD_SIZE = 64

# Each cell of the phenotype is represented with a single unsigned byte. We're
# using boolean states, but for convenience, use the RGB hex codes for black
# (ALIVE) and white (DEAD).
ALIVE = 0x00
DEAD = 0xFF

# Max number of threads per block on a 8.6cc GPU
MAX_THREADS_PER_BLOCK = 1024
# Break up all the rows into groups of NUM_ROW_GROUPS
COL_BATCH_SIZE = MAX_THREADS_PER_BLOCK // WORLD_SIZE
# Number of rows computed by each thread.
COLS_PER_THREAD = WORLD_SIZE // COL_BATCH_SIZE

# Neighbors per layer: L0 + L1 + L2
LAYER_0_NUM_WEIGHTS =   9 +  1 +  0
LAYER_1_NUM_WEIGHTS =   4 +  9 +  1
LAYER_2_NUM_WEIGHTS =   0 +  4 +  9
TOTAL_WEIGHTS = (
    LAYER_0_NUM_WEIGHTS + LAYER_1_NUM_WEIGHTS + LAYER_2_NUM_WEIGHTS)

# Start indices for weights applied to each layer.
LAYER_0_WEIGHTS_START = 0
LAYER_1_WEIGHTS_START = LAYER_0_WEIGHTS_START + LAYER_0_NUM_WEIGHTS
LAYER_2_WEIGHTS_START = LAYER_1_WEIGHTS_START + LAYER_1_NUM_WEIGHTS


@cuda.jit
def update_layer2(phenotypes, weights, population_index, step, row, col):
    """Compute the next state for a single cell in layer2 from prev states."""
    # "Granularity" for computing neighbors.
    g = 4

    # For every cell in this cell's neighborhood there is a unique index into
    # the weight array.
    weight_index = LAYER_2_WEIGHTS_START

    # Next state is computed from the sum of each neighbor cell times its
    # corresponding weight from the weights array.
    next_state = 0.0

    # Don't look at layer0.

    # Look at all the cells that occupy "the same position" as this cell
    # in layer1 ("downward neighbors")
    for r in range((row // g)*g, (row // g)*g + g):
        for c in range((col // g)*g, (col // g)*g + g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            prev_state = phenotypes[population_index][step-1][1][r][c]
            next_state += prev_state * weights[population_index][weight_index]
            weight_index += 1

    # Look at layer2 state in a Moore neighborhood around (row, col)
    for r in range(row-g, row+g+1, g):
        for c in range(col-g, col+g+1, g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            prev_state = phenotypes[population_index][step-1][2][r][c]
            next_state += prev_state * weights[population_index][weight_index]
            weight_index += 1

    # Actually update the phenotype state for step on layer1 at (row, col).
    phenotypes[population_index][step][2][row][col] = (
        # Normalize state to either DEAD or ALIVE.
        ALIVE if next_state < 128 else DEAD)


@cuda.jit
def update_layer1(phenotypes, weights, population_index, step, row, col):
    """Compute the next state for a single cell in layer1 from prev states."""
    # "Granularity" for computing neighbors.
    g = 2

    # For every cell in this cell's neighborhood there is a unique index into
    # the weight array.
    weight_index = LAYER_1_WEIGHTS_START

    # Next state is computed from the sum of each neighbor cell times its
    # corresponding weight from the weights array.
    next_state = 0.0

    # Look at all the cells that occupy "the same position" as this cell
    # in layer0 ("downward neighbors")
    for r in range((row // g)*g, (row // g)*g + g):
        for c in range((col // g)*g, (col // g)*g + g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            prev_state = phenotypes[population_index][step-1][0][r][c]
            next_state += prev_state * weights[population_index][weight_index]
            weight_index += 1

    # Look at layer1 state in a Moore neighborhood around (row, col)
    for r in range(row-g, row+g+1, g):
        for c in range(col-g, col+g+1, g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            prev_state = phenotypes[population_index][step-1][1][r][c]
            next_state += prev_state * weights[population_index][weight_index]
            weight_index += 1

    # Look at the layer2 state at this position (an "upward neighbor")
    prev_state = phenotypes[population_index][step-1][2][row][col]
    next_state += prev_state * weights[population_index][weight_index]

    # Actually update the phenotype state for step on layer1 at (row, col).
    phenotypes[population_index][step][1][row][col] = (
        # Normalize state to either DEAD or ALIVE.
        ALIVE if next_state < 128 else DEAD)


@cuda.jit
def update_layer0(phenotypes, weights, population_index, step, row, col):
    """Compute the next state for a single cell in layer0 from prev states."""
    # Bounds checking: there's a one-cell-wide border of static DEAD cells.
    if (row == 0 or row == WORLD_SIZE - 1 or
        col == 0 or col == WORLD_SIZE - 1):
        phenotypes[population_index][step][0][row][col] = DEAD

    # For every cell in this cell's neighborhood there is a unique index into
    # the weight array.
    weight_index = LAYER_0_WEIGHTS_START

    # Next state is computed from the sum of each neighbor cell times its
    # corresponding weight from the weights array.
    next_state = 0.0

    # Look at layer0 state in a Moore neighborhood around (row, col)
    for r in range(row-1, row+2):
        for c in range(col-1, col+2):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            prev_state = phenotypes[population_index][step-1][0][r][c]
            next_state += prev_state * weights[population_index][weight_index]
            weight_index += 1

    # Look at the layer1 state at this position (an "upward neighbor")
    prev_state = phenotypes[population_index][step-1][1][row][col]
    next_state += prev_state * weights[population_index][weight_index]

    # Don't look at layer2.

    # Actually update the phenotype state for step on layer0 at (row, col).
    phenotypes[population_index][step][0][row][col] = (
        # Normalize state to either DEAD or ALIVE.
        ALIVE if next_state < 128 else DEAD)


@cuda.jit
def simulation_kernel(genotypes, max_depths, phenotypes):
    """Compute and record the full development process of a population."""
    # Compute indices for this thread.
    population_index = cuda.blockIdx.x
    row = cuda.threadIdx.x
    start_col = cuda.threadIdx.y * COLS_PER_THREAD

    # For each step in the simulation (state at step 0 is pre-populated by the
    # caller)...
    for step in range(1, NUM_STEPS):
        # This thread will compute COLS_PER_THREAD contiguous cells from row
        # starting at start_col.
        for col in range(start_col, start_col + COLS_PER_THREAD):
            # Update the state in layer0
            update_layer0(
                phenotypes, genotypes, population_index, step, row, col)
            # If this individual uses layer1, update that, too.
            if max_depths[population_index] > 0:
                update_layer1(
                    phenotypes, genotypes, population_index, step, row, col)
            # If this individual uses layer2, update that, too.
            if max_depths[population_index] > 1:
                update_layer2(
                    phenotypes, genotypes, population_index, step, row, col)
        # Make sure we completely finish computing one step before going on to
        # compute the next step.
        cuda.syncthreads()


def check_granularity(N, image):
    """Returns True iff image is tilable with NxN squares of a single value."""
    # Scale down by sampling every Nth cell in both dimensions.
    scaled_down = image[0::N, 0::N]
    # Scale back up by repeating every cell N times in both dimensions.
    scaled_up = np.repeat(np.repeat(scaled_down, N, 0), N, 1)
    # Check whether the original image matches the resampled version.
    return image == scaled_up


def simulate(genotypes, depths):
    """Simulate genotypes and return phenotype videos."""
    # Infer population size from genotypes
    population_size = genotypes.shape[0]

    # Each individual has a genotype, which consists of a set of TOTAL_WEIGHTS
    # 16-bit floating point values which define the CA update rules.
    assert genotypes.shape == (population_size, TOTAL_WEIGHTS)
    assert genotypes.dtype == np.float16

    # Each individual is configured with a depth of 0, 1, or 2. This way, we
    # can simulate individuals from control and experiment in the same batch.
    # Individuals with lower depth should complete faster. That could hurt
    # performance, but shouldn't because each individual is handled by
    # MAX_THREADS_PER_BLOCK threads, which should mean that whole warps fall
    # out of the computation together.
    assert depths.shape == (population_size,)
    assert depths.dtype == np.uint8
    assert all(depth in range(MAX_DEPTH) for depth in depths)

    # For each inidividual, capture phenotype development over NUM_STEPS. Each
    # phenotype has MAX_DEPTH layers which are all WORLD_SIZE x WORLD_SIZE
    # squares. Layer0 is the "ground truth" of the CA while layers 1 and 2
    # represent a sort of hierarchical internal state for the organism. Layers
    # 1 and 2 are conceptually smaller than layer0 (1/4 and 1/8 respectively),
    # but are represented using arrays of the same size for simplicity.
    phenotypes = np.full(
        (population_size, NUM_STEPS, MAX_DEPTH, WORLD_SIZE, WORLD_SIZE),
        DEAD, dtype=np.uint8)

    # Use a single ALIVE pixel in the middle of the CA world as the initial
    # phenotype state for all individuals in the population.
    for i in range(population_size):
        middle = WORLD_SIZE // 2
        phenotypes[i][0][0][middle][middle] = ALIVE

    # Copy input data from host memory to device memory.
    d_phenotypes = cuda.to_device(phenotypes)
    d_genotypes = cuda.to_device(genotypes)
    d_depths = cuda.to_device(depths)

    # Actually run the simulation for all individuals in parallel on the GPU.
    simulation_kernel[
        # Each grid contains one block per organism.
        (population_size,),
        # Organize threads into two dimensions. Each thread computes
        # COLS_PER_THREAD cells in the CA. The X dimension is the row within
        # the CA world to compute, and the Y dimension is multiplied by
        # COLS_PER_THREAD to find the first column to start from.
        (WORLD_SIZE, COL_BATCH_SIZE)
    ](d_genotypes, d_depths, d_phenotypes)

    # Copy output data from device memory to host memory.
    phenotypes = d_phenotypes.copy_to_host()

    # Layer1 in all phenotypes from all steps of the simulation has a
    # granularity of 2x2.
    assert all(check_granularity(2, p)
               for p in np.reshape(
                   phenotypes[:, :, 1], (-1, WORLD_SIZE, WORLD_SIZE)))

    # Layer2 in all phenotypes from all steps of the simulation has a
    # granularity of 4x4.
    assert all(check_granularity(4, p)
               for p in np.reshape(
                   phenotypes[:, :, 2], (-1, WORLD_SIZE, WORLD_SIZE)))

    return phenotypes


def compute_fitness(phenotypes, target):
    """Score a set of phenotypes generated by the simulate function."""
    # Infer population_size from phenotypes
    population_size = phenotypes.shape[0]
    # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
    assert phenotypes.shape == (population_size, WORLD_SIZE, WORLD_SIZE)
    assert target.shape == (WORLD_SIZE, WORLD_SIZE)

    # Allocate space for results.
    fitness_scores = np.zeros(population_size)

    # For each individual in the population...
    for i in range(population_size):
        # Look at just the final state of the layer0 part of the phenotype.
        # Compare it to the target image, and sum up the deltas to get the
        # final fitness score (lower is better, 0 is a perfect score).
        fitness_scores[i] = np.sum(np.abs(target - phenotypes[i][-1][0]))

    return fitness_scores



def visualize(phenotype, filename=None):
    layer0, layer1, layer3 = phenotype
    merged = (layer0 + layer1 + layer3) // 3
    image = Image.fromarray(merged)
    if filename:
        image.save(filename)
    else:
        image.show()


def demo():
    """Run a simple demo to sanity check this code."""
    population_size = 9

    # Randomize weights to values in the range [-1.0, 1.0)
    genotypes = np.array(
        np.random.random((population_size, TOTAL_WEIGHTS)) * 2 - 1,
        dtype=np.float16)

    # Set an equal number of individuals to have depth 0, 1, and 2.
    depths = np.hstack((
        np.full(population_size // 3, 0, dtype=np.uint8),
        np.full(population_size // 3, 1, dtype=np.uint8),
        np.full(population_size // 3, 2, dtype=np.uint8)))

    # Actually run the simulations, and time how long it takes.
    start = time.perf_counter()
    phenotypes = simulate(genotypes, depths)
    elapsed = time.perf_counter() - start
    print(f'Finished all simulations in {elapsed:0.2f} seconds.')

    # Compute fitness just to run the function through its paces. Both the
    # genotypes and the target are random, so the result is meaningless.
    target = np.random.randint(0, 1, (WORLD_SIZE, WORLD_SIZE)) * DEAD
    compute_fitness(phenotypes, target)

    # Output a gif video for the development for the whole population.
    for i, phenotype in enumerate(phenotypes):
        visualize(phenotype, f'phenotype{i}.gif')


if __name__ == '__main__':
    demo()
