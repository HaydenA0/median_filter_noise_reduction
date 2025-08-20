import numpy as np


def adaptive_median_filter(noisy_image, initial_size=3, max_size=7):
    if not isinstance(noisy_image, np.ndarray) or noisy_image.ndim != 2:
        raise TypeError("Input must be a 2D numpy (grayscale) array.")

    result = np.zeros_like(noisy_image)
    rows, cols = noisy_image.shape

    pad_width = max_size // 2
    padded_image = np.pad(noisy_image, pad_width, mode="symmetric")

    for x in range(rows):
        for y in range(cols):
            window_size = initial_size
            while window_size <= max_size:
                px, py = x + pad_width, y + pad_width

                k = window_size // 2
                window = padded_image[px - k : px + k + 1, py - k : py + k + 1]

                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)

                if z_min < z_med < z_max:
                    z_xy = noisy_image[x, y]
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
    return result.astype(noisy_image.dtype)
