import numpy as np
from numba import njit


def adaptive_median_filter_wrapper(noisy_image, initial_size=3, max_size=7):
    """
    Wrapper function: validate input and pad image before Numba processing.
    """
    if not isinstance(noisy_image, np.ndarray) or noisy_image.ndim != 2:
        raise TypeError("Input must be a 2D numpy (grayscale) array.")

    pad_width = max_size // 2
    padded_image = np.pad(noisy_image, pad_width, mode="symmetric")

    result = adaptive_median_filter_numba(
        padded_image, noisy_image.shape, initial_size, max_size
    )
    return result.astype(noisy_image.dtype)


@njit  # Numba JIT compilation
def adaptive_median_filter_numba(padded_image, orig_shape, initial_size, max_size):
    rows, cols = orig_shape
    result = np.zeros(orig_shape, dtype=padded_image.dtype)
    pad_width = max_size // 2

    for x in range(rows):
        for y in range(cols):
            window_size = initial_size
            while window_size <= max_size:
                px, py = x + pad_width, y + pad_width
                k = window_size // 2
                # Extract window manually
                window = padded_image[px - k : px + k + 1, py - k : py + k + 1].ravel()

                # Manual median (Numba-compatible)
                sorted_window = np.sort(window)
                z_med = sorted_window[len(sorted_window) // 2]
                z_min = sorted_window[0]
                z_max = sorted_window[-1]
                z_xy = padded_image[px, py]

                if z_min < z_med < z_max:
                    if z_min < z_xy < z_max:
                        result[x, y] = z_xy
                    else:
                        result[x, y] = z_med
                    break
                else:
                    window_size += 2
                    if window_size > max_size:
                        result[x, y] = z_med
                        break
    return result
