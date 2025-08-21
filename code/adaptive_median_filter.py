import numpy as np
from numba import njit


@njit
def adaptive_median_filter_vec_compiled(noisy_image, initial_size=3, max_size=7):
    rows, cols = noisy_image.shape
    pad_width = max_size // 2
    padded_image = np.zeros(
        (rows + 2 * pad_width, cols + 2 * pad_width), dtype=noisy_image.dtype
    )

    # Symmetric padding
    for i in range(rows):
        for j in range(cols):
            padded_image[i + pad_width, j + pad_width] = noisy_image[i, j]

    # Top and bottom padding
    for i in range(pad_width):
        for j in range(cols):
            padded_image[i, j + pad_width] = noisy_image[pad_width - i - 1, j]
            padded_image[rows + pad_width + i, j + pad_width] = noisy_image[
                rows - i - 1, j
            ]

    # Left and right padding
    for i in range(rows + 2 * pad_width):
        for j in range(pad_width):
            padded_image[i, j] = padded_image[i, 2 * pad_width - j - 1]
            padded_image[i, cols + pad_width + j] = padded_image[
                i, cols + pad_width - j - 1
            ]

    result = np.zeros((rows, cols), dtype=noisy_image.dtype)
    processed_mask = np.zeros((rows, cols), dtype=np.bool_)

    for size in range(initial_size, max_size + 2, 2):
        k = size // 2

        for i in range(rows):
            for j in range(cols):
                if processed_mask[i, j]:
                    continue

                # Extract window
                window = padded_image[
                    i + pad_width - k : i + pad_width + k + 1,
                    j + pad_width - k : j + pad_width + k + 1,
                ]

                # Compute statistics
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = noisy_image[i, j]

                stage_A_passed = (z_med > z_min) and (z_med < z_max)
                if stage_A_passed:
                    if (z_xy > z_min) and (z_xy < z_max):
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    processed_mask[i, j] = True

        if np.all(processed_mask):
            break

    # Fill remaining pixels with median of largest window
    for i in range(rows):
        for j in range(cols):
            if not processed_mask[i, j]:
                k = max_size // 2
                window = padded_image[
                    i + pad_width - k : i + pad_width + k + 1,
                    j + pad_width - k : j + pad_width + k + 1,
                ]
                result[i, j] = np.median(window)

    return result
