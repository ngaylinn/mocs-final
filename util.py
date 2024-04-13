import numpy as np

def create_complex(N):
    if N <= 6 or N % 2 != 0:
        raise ValueError("N must be an even number greater than 6")

    shape = np.array([
        [0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,0,0],
        [0,1,1,0,0,1,1,0],
        [0,0,0,0,0,1,0,0],
        [0,1,1,1,0,1,0,0],
        [0,1,0,1,1,1,1,0],
        [0,1,1,1,0,1,0,0],
        [0,0,0,0,0,0,0,0],
    ])

    if N == 8:
        return shape
    if N == 64:
        return np.repeat(np.repeat(shape,8, axis=0),8, axis=1)

    return shape 
    


def create_hollow_circle(N):
    if N <= 6 or N % 2 != 0:
        raise ValueError("N must be an even number greater than 6")

    # Create an N x N array of zeros
    array = np.zeros((N, N))

    # Calculate the 2x2 center square and the radius
    center_x1 = N // 2 - 1
    center_y1 = N // 2 - 1
    center_x2 = N // 2
    center_y2 = N // 2

    # radius_outer = N // 2.5
    # radius_inner = N // 5

    radius_outer = N / 3 
    radius_inner = N / 8

    # Fill the array with a circle of ones
    for y in range(N):
        for x in range(N):
            if ((x - center_x1)**2 + (y - center_y1)**2 < radius_outer**2 or
                (x - center_x2)**2 + (y - center_y1)**2 < radius_outer**2 or
                (x - center_x1)**2 + (y - center_y2)**2 < radius_outer**2 or
                (x - center_x2)**2 + (y - center_y2)**2 < radius_outer**2):
                array[y, x] = 1

    for y in range(N):
        for x in range(N):
            if ((x - center_x1)**2 + (y - center_y1)**2 < radius_inner**2 or
                (x - center_x2)**2 + (y - center_y1)**2 < radius_inner**2 or
                (x - center_x1)**2 + (y - center_y2)**2 < radius_inner**2 or
                (x - center_x2)**2 + (y - center_y2)**2 < radius_inner**2):
                array[y, x] = 0

    return array

def create_diamond(N):
    # Create an N x N array of zeros
    array = np.zeros((N, N))

    # Calculate the buffer size based on N (e.g., 1/4th of N)
    buffer_size = N // 4

    # Adjusted size for the diamond calculation
    adjusted_size = N - 2 * buffer_size

    # Calculate the middle index (for the center of the diamond)
    mid = adjusted_size // 2 + buffer_size

    # Fill the array with a diamond of ones, considering the buffer
    for y in range(buffer_size, N - buffer_size + 1):
        for x in range(buffer_size, N - buffer_size + 1):
            # Calculate the absolute distance from the center, adjusted for buffer
            dist = abs(x - mid) + abs(y - mid)

            # Fill with 1 if the distance is less than or equal to mid (for diamond shape)
            if dist <= mid - buffer_size:
                array[y, x] = 1

    return array

def create_square(N):
    array = np.full((N, N), 0)
    array[(N // 4):(N//4 * 3), (N // 4):(N//4 * 3)] = 1

    return array

def create_plus(N):
    array = np.full((8,8), 0)
    array[3:5, 1:7] = 1
    array[1:7, 3:5] = 1
    array = np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)

    return array
