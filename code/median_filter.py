import numpy as np


def medians_grid(grid):
    try:
        red_values = [pixel[0] for pixel in grid]
        green_values = [pixel[1] for pixel in grid]
        blue_values = [pixel[2] for pixel in grid]
        return (np.median(red_values), np.median(green_values), np.median(blue_values))
    except Exception as e:
        raise ValueError(f"Error while computing RGB medians: {e}")


def generate_indicies_offsets(n: int):
    if n % 2 == 0 or n < 1:
        raise ValueError("Grid size must be a positive odd integer.")
    k = n // 2
    part_negative = list(range(-k, 0))
    part_positive = list(range(0, k + 1))
    return part_negative + part_positive


def list_extract_grid(noisy_image, x, y, grid_size):
    try:
        grid = []
        offsets = generate_indicies_offsets(grid_size)
        for offset_y in offsets:
            for offset_x in offsets:
                grid.append(noisy_image[x + offset_y][y + offset_x])
        return grid
    except IndexError:
        raise IndexError("Grid extraction went out of image bounds.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in list_extract_grid: {e}")


def median_filter(noisy_image, grid_size: int = 3):
    if not isinstance(noisy_image, np.ndarray):
        raise TypeError("Input noisy_image must be a numpy array.")

    if noisy_image.ndim not in [2, 3]:
        raise ValueError("Image must be either 2D (grayscale) or 3D (RGB).")

    try:
        result = noisy_image.copy()
        k = grid_size // 2

        for x in range(k, noisy_image.shape[0] - k):
            for y in range(k, noisy_image.shape[1] - k):
                grid = list_extract_grid(noisy_image, x, y, grid_size)

                if noisy_image.ndim == 2:  # grayscale
                    grid.sort()
                    median = grid[len(grid) // 2]
                    result[x, y] = median
                else:  # RGB
                    result[x, y] = medians_grid(grid)

        return result
    except Exception as e:
        raise RuntimeError(f"Error while applying median filter: {e}")
