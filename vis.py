import pickle
import time
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from afpo import Solution, activation2int
from simulation import simulate, make_seed_phenotypes, DEAD, ALIVE


def simulate_one_individual(solution : Solution):
    init_phenotypes = make_seed_phenotypes(1, n_layers=solution.n_layers)
    print(solution.n_layers)

    phenotypes = simulate(
            np.array([solution.growth_genotype]), 
            np.array([solution.state_genotype]), 
            solution.n_layers, 
            solution.base_layer,  
            solution.around_start, 
            solution.above_start, 
            True, 
            init_phenotypes, 
            0) # sigmoid default currently 
    
    return phenotypes[0]

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
        total_width = w * 4

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

    # print(exp.shape)
    # print(exp.get_target_shape())

    # exp_best_phenotype = simulate_one_individual(exp.best_solution())
    # visualize_all_layers(exp_best_phenotype, 'control_best_all_layers_1.gif')
    exp_best_phenotype = simulate_one_individual(exp.best_solution())
    save_folder = '/'.join(args.exp.split('/')[:-2]) + '/vis'
    file_name = args.exp.split('/')[-1]
    os.makedirs(f'{save_folder}', exist_ok=True)

    visualize_all_layers(exp_best_phenotype, f'{save_folder}/{file_name}.gif', base_layer_idx=exp.base_layer)
