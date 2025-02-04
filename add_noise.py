import pickle
import time
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageOps, ImageEnhance

from optimizers.afpo import Solution, AgeFitnessPareto, activation2int
from simulation import simulate, make_seed_phenotypes, DEAD, ALIVE, make_seed_phenotypes_layer, NUM_STEPS, WORLD_SIZE

color_dict = {
    'green': {
        'live_inside': 0xFF00AA00,        # Vibrant green for live_inside
        'dead_inside': 0xFF77FF77,       # Slightly less vibrant green for dead_inside
        'live_outside': 0xFF006600,     # Dark green for live_outside
        'dead_outside': 0xFFEEFFEE,   # Almost white, but not quite, for dead_outside
    },
    'red': {
        'live_inside': 0xFF0000FF,        # Vibrant red for live_inside
        'dead_inside': 0xFFBBBBFF,       # Slightly less vibrant red for dead_inside
        'live_outside': 0xFF000066,     # Dark red for live_outside
        'dead_outside': 0xFFEEEEFF,   # Almost white, but not quite, for dead_outside
    },
    'blue': {
        'live_inside': 0xFFFF0000,        # Vibrant red for live_inside
        'dead_inside': 0xFFFFBBBB,       # Slightly less vibrant red for dead_inside
        'live_outside': 0xFF660000,     # Dark red for live_outside
        'dead_outside': 0xFFFFEEEE,   # Almost white, but not quite, for dead_outside
    }
}


def visualize_layer_over_time(phenotype, filename, layer_idx, target_shape, frames = [0, 5, 10, 50, 99]):

    n_layers, l, w = phenotype[frames[0]].shape

    assert layer_idx in range(n_layers)

    # Set the desired gap size between panels (you can adjust this value)
    gap_size = 5  # Adjust this value as needed

    # Calculate the total width for the new image with gaps
    total_width = w * len(frames) + gap_size * (len(frames) - 1) + len(frames)
    total_height = l + gap_size * 2 + len(frames)

    # Create a new image with the calculated total width
    combined_image = Image.new('RGBA', (total_width, total_height))

    # Add a panel for each frame in frames
    for i, frame_data in enumerate(phenotype[frames]):
        layer_data = frame_data[layer_idx]

        img = Image.fromarray(layer_data, mode='RGBA')

        # Increase opacity to full
        datas = img.getdata()
        newData = []
        for item in datas:
            newData.append(item[:3] + (255,))  # Change alpha to 255
        img.putdata(newData)
        
        img_enhanced = ImageOps.expand(img, border=1, fill=(0, 0, 0, 255))
        combined_image.paste(img_enhanced, (i * (w + gap_size), gap_size))

        # img = ImageOps.expand(img, border=1, fill=(0, 0, 0, 255))
        # combined_image.paste(img, (i * (w + gap_size), gap_size))

    combined_image.save(filename)

def visualize_layers_and_selection_over_time(phenotype, filename, base_layer_idx, target_shape, color, frames=[0, 5, 10, 50, 99]):
    # Frame indices

    n_layers, l, w = phenotype[frames[0]].shape

    # Set the desired gap size between panels (you can adjust this value)
    gap_size = 5  # Adjust this value as needed

    # Calculate the total width for the new image with gaps
    total_width = w * len(frames) + gap_size * (len(frames) - 1) + len(frames)
    total_height = (n_layers+1)*l + gap_size * n_layers + len(frames)

    # Create a new image with the calculated total width
    combined_image = Image.new('RGBA', (total_width, total_height))

    for i, frame_data in enumerate(phenotype[frames]):
        base_layer = frame_data[base_layer_idx]

        base = np.array(
            np.bitwise_or(
                np.bitwise_or(
                    (np.bitwise_and((base_layer != DEAD), (target_shape != DEAD))) * color_dict[color]['live_inside'],   # live correct 
                    (np.bitwise_and((base_layer == DEAD), (target_shape != DEAD))) * color_dict[color]['dead_inside']),
                np.bitwise_or(
                    (np.bitwise_and((base_layer != DEAD), (target_shape == DEAD))) * color_dict[color]['live_outside'],
                    (np.bitwise_and((base_layer == DEAD), (target_shape == DEAD))) * color_dict[color]['dead_outside'])
                ), # ALIVE cells are white
            dtype=np.uint32)

        # Base layer first
        base_img = Image.fromarray(base, mode='RGBA')
        base_img = ImageOps.expand(base_img, border=1, fill=(0, 0, 0, 255))
        combined_image.paste(base_img, (i * (w + gap_size), gap_size))

        # Now add the other layers
        for layer_idx in reversed(range(n_layers)):
            layer_data = frame_data[layer_idx]
            img = Image.fromarray(layer_data, mode='RGBA')

            # Increase opacity to full
            datas = img.getdata()
            newData = []
            for item in datas:
                newData.append(item[:3] + (255,))  # Change alpha to 255
            img.putdata(newData)
            
            img_enhanced = ImageOps.expand(img, border=1, fill=(0, 0, 0, 255))
            combined_image.paste(img_enhanced, (i * (w + gap_size), gap_size + ( n_layers-layer_idx) * (l + gap_size)))

    combined_image.save(filename)

def simulate_one_individual(afpo, solution : Solution):
    init_phenotypes = make_seed_phenotypes_layer(1, n_layers=afpo.n_layers, base_layer=afpo.base_layer)
    print(solution.n_layers)

    phenotypes = simulate(
            np.array([solution.state_genotype]),
            solution.n_layers,  
            solution.around_start, 
            solution.above_start, 
            phenotypes=init_phenotypes,
            below_map=afpo.below_map,
            above_map=afpo.above_map)
    
    return phenotypes[0]

def simulate_one_individual_noisy(afpo, solution : Solution, noise):
    init_phenotypes = make_seed_phenotypes_layer(1, n_layers=afpo.n_layers, base_layer=afpo.base_layer)

    phenotypes = simulate(
            np.array([solution.state_genotype]),
            solution.n_layers,  
            solution.around_start, 
            solution.above_start, 
            phenotypes=init_phenotypes,
            below_map=afpo.below_map,
            above_map=afpo.above_map,
            noise=noise)
    
    return phenotypes[0]

def simulate_n_individuals(solutions):
    n = len(solutions)
    n_layers = solutions[0].n_layers
    base_layer = solutions[0].base_layer
    around_start = solutions[0].around_start
    above_start = solutions[0].above_start
    init_phenotypes = make_seed_phenotypes(n, n_layers)

    phenotypes = simulate(
            np.array([soln.growth_genotype for soln in solutions]), 
            np.array([soln.state_genotype for soln in solutions]), 
            n_layers, 
            base_layer,  
            around_start, 
            above_start, 
            True, 
            init_phenotypes, 
            0) # sigmoid default currently 
    
    return phenotypes[:,-1,:,:,:]

def visualize_all_layers(phenotype, filename, base_layer_idx=0):
    def make_frame(frame_data):
        l, w = frame_data[0].shape
        base = np.array(
            np.bitwise_or(
                (frame_data[0] != DEAD) * 0xffffffff,   # DEAD cells are black
                (frame_data[0] == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)

        # Calculate the total width for the new image
        n_layers = frame_data.shape[0]
        total_width = w * (n_layers + 1)

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        # Convert layers to images and paste in combined image
        base_image = Image.fromarray(base, mode='RGBA')
        combined_image.paste(base_image, (0, 0))
        width_offset = w
        for layer in frame_data:
            image = Image.fromarray(layer, mode='RGBA')
            combined_image.paste(image, (w, 0))
            w += w

        return combined_image

    '''
    def make_frame(frame_data):
        n_layers, l, w = frame_data.shape
        
        base_layer = frame_data[base_layer_idx]

        base = np.array(
            np.bitwise_or(
                (base_layer != DEAD) * 0xffffffff,   # DEAD cells are black
                (base_layer == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)

        # Calculate the total width for the new image
        total_width = w * 5

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        # Base layer first
        base_img = Image.fromarray(base, mode='RGBA')
        combined_image.paste(base_img, (0, 0))

        # Convert layers to images
        for layer in range(n_layers):
            img = Image.fromarray(frame_data[layer], mode='RGBA')
            combined_image.paste(img, ((layer+1) * w, 0))

        return combined_image
    '''

    frames = [make_frame(frame_data) for frame_data in phenotype]
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)



def visualize_frames(phenotype, filename, n_frames=4, layer=0):
    def make_image(frames, n_frames, layer):
        l, w = frames[0][0].shape

        # Calculate the total width for the new image
        total_width = w * n_frames

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        for f in range(n_frames):
            layer0, layer1, layer2 = frames[f]
            l, w = layer0.shape

            if layer == 'base':
                base = np.array(
                np.bitwise_or(
                    (layer0 != DEAD) * 0xffffffff,   # DEAD cells are black
                    (layer0 == DEAD) * 0xff000000), # ALIVE cells are white
                dtype=np.uint32)
                img = Image.fromarray(base, mode='RGBA')
            else:
                img = Image.fromarray(frames[f][layer], mode='RGBA')

            combined_image.paste(img, (w*f, 0))

        return combined_image

    img = make_image(phenotype, n_frames, layer)
    
    img.save(filename)



def visualize_one_layer(phenotype, filename, layer=0):
    def make_frame(frame_data):
        # Scale up the image 4x to make it more visible.
        # frame_data = frame_data.repeat(4, 1).repeat(4, 2)
        layer0, layer1, layer2, layer3 = frame_data
        l, w = layer0.shape
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
        elif layer == 3:
            return Image.fromarray(layer2, mode='RGBA') 
        elif layer == 'base':
            return Image.fromarray(base, mode='RGBA')


    frames = [make_frame(frame_data) for frame_data in phenotype]
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)


def visualize_all_layers_timestep(phenotype, filename, timestep):
    print('Shape of phenotype: ', phenotype.shape)
    def make_frame(frame_data):
        l, w = frame_data[0].shape
        base = np.array(
            np.bitwise_or(
                (frame_data[0] != DEAD) * 0xffffffff,   # DEAD cells are black
                (frame_data[0] == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)

        # Calculate the total width for the new image
        n_layers = frame_data.shape[0]
        total_width = w * (n_layers + 1)

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        # Convert layers to images and paste in combined image
        base_image = Image.fromarray(base, mode='RGBA')
        combined_image.paste(base_image, (0, 0))
        width_offset = w
        for layer in frame_data:
            image = Image.fromarray(layer, mode='RGBA')
            combined_image.paste(image, (w, 0))
            w += w

        return combined_image

    frame = make_frame(phenotype[timestep]) 
    
    frame.save(filename)

def generate_knockouts_radiation(T,L,N, phenotype, timestep_range, layers, p):
    noise = generate_empty_noise(T,L,N)

    for timestep in range(*timestep_range):
        for l in layers:
            for r in range(WORLD_SIZE):
                for c in range(WORLD_SIZE):
                    if np.random.random() < p:
                        noise[timestep, l, r, c] = 1

    return noise
          
    

def generate_empty_noise(T, L, N):
    return np.zeros((T, L, N, N))

def generate_noise_knockout(T, L, N, knockout_locations):
    """
    knockout_locations is a list of tuples, (t, r, c) where
    t is the timestep, and r,c is the row and column of the knockout
    """
    noise = generate_empty_noise(T,L,N)
    for (t, l, r, c) in knockout_locations: 
        noise[t,l,r,c] = 1

    return noise

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default=None)
args = parser.parse_args()

experiment_pkl = args.exp

if __name__ == '__main__':
    with open(experiment_pkl, 'rb') as pf:
        exp = pickle.load(pf)

    best = exp.best_solution()
    print('Fitness: ', best.fitness)
    print(best.state_genotype)

    n_timesteps = NUM_STEPS
    n_layers = best.n_layers

    '''
    Generate the normal NCA (w/ no noise)
    '''
    empty_noise = generate_empty_noise(n_timesteps, n_layers, WORLD_SIZE)
    exp_best_phenotype = simulate_one_individual_noisy(exp, best, empty_noise)

    '''
    Add noise to the model!
    '''
    # noise = generate_knockouts_radiation(n_timesteps, n_layers, WORLD_SIZE, exp_best_phenotype, timestep_range=(25,75), layers=[0], p=0.001)

    timestep_1 = 50
    timestep_2 = 80
    poke_x, poke_y = 43, 27
    knockouts = [(timestep_1, 0, poke_x, poke_y), (timestep_2, 0, poke_x, poke_y)]
    knockouts = knockouts + [(timestep_1, 0, poke_x+1, poke_y), (timestep_1, 0, poke_x-1, poke_y),(timestep_1, 0, poke_x, poke_y+1), (timestep_1, 0, poke_x, poke_y-1)]
    knockouts = knockouts + [(timestep_2, 0, poke_x+1, poke_y), (timestep_2, 0, poke_x-1, poke_y),(timestep_2, 0, poke_x, poke_y+1), (timestep_2, 0, poke_x, poke_y-1)]
    knockouts = knockouts + [(timestep_1, 0, poke_x+1, poke_y+1), (timestep_1, 0, poke_x-1, poke_y-1),(timestep_1, 0, poke_x-1, poke_y+1), (timestep_1, 0, poke_x+1, poke_y-1)]
    knockouts = knockouts + [(timestep_2, 0, poke_x+1, poke_y+1), (timestep_2, 0, poke_x-1, poke_y-1),(timestep_2, 0, poke_x-1, poke_y+1), (timestep_2, 0, poke_x+1, poke_y-1)]

    noise = generate_noise_knockout(n_timesteps, n_layers, WORLD_SIZE, knockouts)
    exp_best_phenotype_noise = simulate_one_individual_noisy(exp, best, noise)

    visualize_all_layers_timestep(exp_best_phenotype_noise, f'./vis/poke_recovery/timestep{timestep_1-1}.png', timestep=timestep_1-1)
    visualize_all_layers_timestep(exp_best_phenotype_noise, f'./vis/poke_recovery/timestep{timestep_2-1}.png', timestep=timestep_2-1)
    for i in range(14):
        visualize_all_layers_timestep(exp_best_phenotype_noise, f'./vis/poke_recovery/timestep{timestep_1+i}.png', timestep=timestep_1+i)
        visualize_all_layers_timestep(exp_best_phenotype_noise, f'./vis/poke_recovery/timestep{timestep_2+i}.png', timestep=timestep_2+i)

    # print(sum(exp_best_phenotype[-1, 3] > 0))
    visualize_all_layers(exp_best_phenotype, './vis/homeostatter.gif', base_layer_idx=exp.base_layer)
    visualize_all_layers(exp_best_phenotype_noise, './vis/homeostatter_noise.gif', base_layer_idx=exp.base_layer)

