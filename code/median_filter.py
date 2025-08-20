import numpy as np


def medians_grid(grid):
    red_values = [pixel[0] for pixel in grid]
    green_value = [pixel[1] for pixel in grid]
    blue_values = [pixel[2] for pixel in grid]
    return np.median(red_values), np.median(green_value), np.median(blue_values)


def generate_indicies_offsets(n: int):
    k = n // 2
    part_negative = list(range(-k, 0))
    part_positive = list(range(0, k + 1))
    return part_negative + part_positive


def list_extract_grid(noisy_image, x, y, grid_size):
    grid = []
    offsets = generate_indicies_offsets(grid_size)
    for offset_y in offsets:
        for offset_x in offsets:
            grid.append(noisy_image[x + offset_y][y + offset_x])
    return grid


def median_filter_bw(noisy_image, grid_size: int = 3):
    result = noisy_image.copy()
    for x in range(1, len(noisy_image) - 1):
        for y in range(1, len(noisy_image[0]) - 1):
            grid = list_extract_grid(noisy_image, x, y, grid_size)
            grid.sort()
            median = grid[len(grid) // 2]
            result[x][y] = median
    return result


def median_filter(noisy_image, grid_size: int = 3):
    result = noisy_image.copy()
    for x in range(grid_size // 2, len(noisy_image) - grid_size // 2):
        for y in range(grid_size // 2, len(noisy_image[0]) - grid_size // 2):
            grid = list_extract_grid(noisy_image, x, y, grid_size)
            result[x][y] = medians_grid(grid)
    return result
