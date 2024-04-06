import pickle
import time
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageOps, ImageEnhance

from afpo import Solution, AgeFitnessPareto, activation2int
from simulation import simulate, make_seed_phenotypes, DEAD, ALIVE, make_seed_phenotypes_layer

# def visualize_selection_layer_over_time(phenotype, filename, base_layer_idx):
#     # Frame indices
#     frames = [0,5,10,50,99]

#     n_layers, l, w = phenotype[frames[0]].shape
    
#     # Calculate the total width for the new image
#     total_width = w * len(frames)

#     # Create a new image with the calculated total width
#     combined_image = Image.new('RGBA', (total_width, l))
    
#     for i, frame_data in enumerate(phenotype[frames]):
#         base_layer = frame_data[base_layer_idx]

#         base = np.array(
#             np.bitwise_or(
#                 (base_layer != DEAD) * 0xffffffff,   # DEAD cells are black
#                 (base_layer == DEAD) * 0xff000000), # ALIVE cells are white
#             dtype=np.uint32)

#         # Base layer first
#         base_img = Image.fromarray(base, mode='RGBA')
#         combined_image.paste(base_img, (i*w, 0))

#     combined_image.save(filename)

# def visualize_selection_layer_over_time(phenotype, filename, base_layer_idx):
#     # Frame indices
#     frames = [0, 5, 10, 50, 99]

#     n_layers, l, w = phenotype[frames[0]].shape

#     # Set the desired gap size between panels (you can adjust this value)
#     gap_size = 5  # Adjust this value as needed

#     # Calculate the total width for the new image with gaps
#     total_width = w * len(frames) + gap_size * (len(frames) - 1) + len(frames)
#     total_height = l + gap_size * 2 + len(frames)

#     # Create a new image with the calculated total width
#     combined_image = Image.new('RGBA', (total_width, total_height))

#     for i, frame_data in enumerate(phenotype[frames]):
#         base_layer = frame_data[base_layer_idx]

#         base = np.array(
#             np.bitwise_or(
#                 (base_layer != DEAD) * 0xffffffff,   # DEAD cells are black
#                 (base_layer == DEAD) * 0xff008000), # ALIVE cells are white
#             dtype=np.uint32)

#         # Base layer first
#         base_img = Image.fromarray(base, mode='RGBA')
#         base_img = ImageOps.expand(base_img, border=1, fill=(0, 0, 0, 255))
#         combined_image.paste(base_img, (i * (w + gap_size), gap_size))

#     combined_image.save(filename)

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

def visualize_selection_layer_over_time(phenotype, filename, base_layer_idx, target_shape, color, frames=[0, 5, 10, 50, 99]):
    # Frame indices

    n_layers, l, w = phenotype[frames[0]].shape

    # Set the desired gap size between panels (you can adjust this value)
    gap_size = 5  # Adjust this value as needed

    # Calculate the total width for the new image with gaps
    total_width = w * len(frames) + gap_size * (len(frames) - 1) + len(frames)
    total_height = l + gap_size * 2 + len(frames)

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

    combined_image.save(filename)

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
# def simulate_one_individual(solution : Solution):
#     init_phenotypes = make_seed_phenotypes(1, n_layers=solution.n_layers)
#     print(solution.n_layers)

#     phenotypes = simulate(
#             np.array([solution.growth_genotype]), 
#             np.array([solution.state_genotype]), 
#             solution.n_layers, 
#             solution.base_layer,  
#             solution.around_start, 
#             solution.above_start, 
#             True, 
#             init_phenotypes, 
#             0) # sigmoid default currently 
    
#     return phenotypes[0]

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
        layer0, layer1, layer2 = frame_data
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
        elif layer == 'base':
            return Image.fromarray(base, mode='RGBA')


    frames = [make_frame(frame_data) for frame_data in phenotype]
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)


def visualize_all_layers_last_timestep(phenotype, filename):
    def make_frame(frame_data):
        layer0, layer1, layer2 = frame_data
        l, w = layer0.shape
        base = np.array(
            np.bitwise_or(
                (layer0 != DEAD) * 0xffffffff,   # DEAD cells are black
                (layer0 == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)

        # Calculate the total width for the new image
        total_width = w * 4

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        # Convert layers to images
        image0 = Image.fromarray(base, mode='RGBA')
        image1 = Image.fromarray(layer0, mode='RGBA')
        image2 = Image.fromarray(layer1, mode='RGBA')
        image3 = Image.fromarray(layer2, mode='RGBA')
        
        # Paste each image side by side in the combined image
        combined_image.paste(image0, (0, 0))
        combined_image.paste(image1, (w, 0))
        combined_image.paste(image2, (w * 2, 0))
        combined_image.paste(image3, (w * 3, 0))

        return combined_image

    frame = make_frame(phenotype[-1]) 
    
    frame.save(filename)


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default=None)
args = parser.parse_args()

experiment_pkl = args.exp


if __name__ == '__main__':
    with open(experiment_pkl, 'rb') as pf:
        exp = pickle.load(pf)

#     # print(exp.shape)
#     # print(exp.get_target_shape())

    best = exp.best_solution()
    # best = exp.parent_population[list(exp.parent_population.keys())[0]]
    print(best.fitness)
    # print(best.n_layers)
    exp_best_phenotype = simulate_one_individual(exp, best)

    # print(sum(exp_best_phenotype[-1, 3] > 0))
    visualize_all_layers(exp_best_phenotype, 'trial.gif', base_layer_idx=exp.base_layer)



    print('hello?')
    
    # exp_best_phenotype = simulate_one_individual(exp, exp.best_solution())
    # print(exp_best_phenotype)
    save_folder = '/'.join(args.exp.split('/')[:-2]) + '/vis'
    file_name = args.exp.split('/')[-1]
#     os.makedirs(f'{save_folder}', exist_ok=True)

#     # visualize_all_layers(exp_best_phenotype, 
#     #                      f'{save_folder}/{file_name}.gif', 
#     #                      base_layer_idx=exp.base_layer)

#     print('total weights: ', exp.best_fitness_history[0].total_weights, exp.best_fitness_history[0].state_n_weights, exp.best_fitness_history[0].growth_n_weights)

    # exp.shape = 'square'
    # visualize_layer_over_time(exp_best_phenotype, 
    #                             f'{save_folder}/{file_name}_state_overtime_l0.png', 
    #                             layer_idx=exp.base_layer,
    #                             target_shape=exp.get_target_shape(),
    #                             )
    # visualize_layer_over_time(exp_best_phenotype, 
    #                             f'{save_folder}/{file_name}_state_overtime_l1.png', 
    #                             layer_idx=exp.base_layer+1,
    #                             target_shape=exp.get_target_shape(),
    #                             )
    # visualize_layer_over_time(exp_best_phenotype, 
    #                             f'{save_folder}/{file_name}_state_overtime_l2.png', 
    #                             layer_idx=exp.base_layer+2,
    #                             target_shape=exp.get_target_shape(),
    #                             )
    # visualize_layer_over_time(exp_best_phenotype, 
    #                             f'{save_folder}/{file_name}_state_overtime_l3.png', 
    #                             layer_idx=exp.base_layer+3,
    #                             target_shape=exp.get_target_shape(),
    #                             )
#     # exp.shape = 'square'
    visualize_selection_layer_over_time(exp_best_phenotype, 
                                        f'{save_folder}/{file_name}_overtime.png', 
                                        base_layer_idx=exp.base_layer,
                                        target_shape=exp.get_target_shape(),
                                        color='red')
