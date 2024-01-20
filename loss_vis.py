import matplotlib.pyplot as plt
import numpy as np

# this script is purely for visualization purposes!!

def draw_initial_ca(grid_size=64, shape_size=32):
    '''Example of the state of the NCA at t=0'''
    # grid of size grid_size x grid_size
    grid = np.zeros((grid_size, grid_size))

    # identify desired shape
    desired_shape_start = (grid_size // 2 - shape_size // 2, grid_size // 2 - shape_size // 2)
    desired_shape_end = (grid_size // 2 + shape_size // 2 - 1, grid_size // 2 + shape_size // 2 - 1)
    grid[desired_shape_start[0]:desired_shape_end[0]+1, desired_shape_start[1]:desired_shape_end[1]+1] = 0.5

    # fill one of the center cells
    initial_cell = (grid_size // 2, grid_size // 2)
    grid[initial_cell] = 1

    # plot stuff
    plt.imshow(grid, cmap='Blues', vmin=0, vmax=1, origin='upper', interpolation='none', aspect='equal')
    plt.title('t=0')
    plt.savefig('initial_state_ex.png')
    plt.show()
    

def draw_final_ca(grid_size=64, shape_size=32):
    '''Example of what the state of the NCA at t=100 might look like'''
    # grid of size grid_size x grid_size
    grid = np.zeros((grid_size, grid_size))

    # identify desired shape
    desired_shape_start = (grid_size // 2 - shape_size // 2, grid_size // 2 - shape_size // 2)
    desired_shape_end = (grid_size // 2 + shape_size // 2 - 1, grid_size // 2 + shape_size // 2 - 1)
    target_shape = np.zeros_like(grid)
    target_shape[desired_shape_start[0]:desired_shape_end[0]+1, desired_shape_start[1]:desired_shape_end[1]+1] = 0.5

    # set a random seed for replicatability 
    np.random.seed(4)

    # choose one of the center cells
    center_cells = [
        (grid_size // 2 - 1, grid_size // 2 - 1),
        (grid_size // 2 - 1, grid_size // 2),
        (grid_size // 2, grid_size // 2 - 1),
        (grid_size // 2, grid_size // 2)
    ]
    current_cell = center_cells[np.random.choice(len(center_cells))]
    grid[current_cell] = 1

    # starting from the center cell, randomly fill 256 cells
    filled_cells_inside = 0
    filled_cells_outside = 0
    total_filled_cells = 1                                                              # Account for initial cell?
    while total_filled_cells < 1024:
        # von Neumann neighborhood (4 neighbors)
        # neighbors = [
        #     ((current_cell[0] - 1) % grid_size, current_cell[1]),
        #     ((current_cell[0] + 1) % grid_size, current_cell[1]),
        #     (current_cell[0], (current_cell[1] - 1) % grid_size),
        #     (current_cell[0], (current_cell[1] + 1) % grid_size)
        # ]

        # moore nieghborhood (8 neighbors)
        neighbors = [
            ((current_cell[0] - 1) % grid_size, (current_cell[1] - 1) % grid_size),
            ((current_cell[0] - 1) % grid_size, current_cell[1]),
            ((current_cell[0] - 1) % grid_size, (current_cell[1] + 1) % grid_size),
            (current_cell[0], (current_cell[1] - 1) % grid_size),
            (current_cell[0], (current_cell[1] + 1) % grid_size),
            ((current_cell[0] + 1) % grid_size, (current_cell[1] - 1) % grid_size),
            ((current_cell[0] + 1) % grid_size, current_cell[1]),
            ((current_cell[0] + 1) % grid_size, (current_cell[1] + 1) % grid_size)
        ]
        # randomly choose a neighbor
        chosen_neighbor = neighbors[np.random.choice(len(neighbors))]

        # check if the chosen neighbor is inside or outside the target shape
        if desired_shape_start[0] <= chosen_neighbor[0] <= desired_shape_end[0] and \
                desired_shape_start[1] <= chosen_neighbor[1] <= desired_shape_end[1]:
            # fill the chosen neighbor cell if inside the target shape
            if grid[chosen_neighbor] == 0:
                grid[chosen_neighbor] = 1
                filled_cells_inside += 1
                total_filled_cells += 1
        else:
            # fill the chosen neighbor cell if outside the target shape
            if grid[chosen_neighbor] == 0:
                grid[chosen_neighbor] = 1
                filled_cells_outside += 1
                total_filled_cells += 1

        # move to the chosen/next neighbor
        current_cell = chosen_neighbor


    # plot stuff
    plt.imshow(target_shape, cmap='Reds', vmin=0, vmax=1, origin='upper', interpolation='none', aspect='equal')
    plt.imshow(grid, cmap='Reds', vmin=0, vmax=1, origin='upper', interpolation='none', aspect='equal', alpha=0.5)
    plt.title('t=100')
    print(f'filled_target_cells = {filled_cells_inside}')
    print(f'filled_nontarget_cells = {filled_cells_outside}')
    plt.savefig('loss_ex.png')
    plt.show()
    




draw_initial_ca()
draw_final_ca()

