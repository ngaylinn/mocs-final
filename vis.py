import pickle
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from afpo import Solution
from simulation import simulate, make_seed_phenotypes, DEAD, ALIVE


def simulate_one_individual(solution : Solution):
    genotypes = np.array([solution.genotype]) 
    layers = np.array([2], dtype=np.uint8)
    phenotypes = make_seed_phenotypes(1)

    phenotypes = simulate(genotypes, layers, phenotypes)
    
    return phenotypes[0]

def visualize_all_layers(phenotype, filename):
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

    frames = [make_frame(frame_data) for frame_data in phenotype]
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)


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
parser.add_argument('exp', type=str)
args = parser.parse_args()

experiment_pkl = args.exp


if __name__ == '__main__':
    with open(experiment_pkl, 'rb') as pf:
        exp = pickle.load(pf)

    exp_best_phenotype = simulate_one_individual(exp.best_solution())
    visualize_all_layers(exp_best_phenotype, 'exp_best_all_layers.png')