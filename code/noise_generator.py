import numpy as np
import random


def add_salt_and_pepper_noise(img_array, noise_probability=0.05):
    try:
        # Validate input
        if not isinstance(img_array, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if not (0 <= noise_probability <= 1):
            raise ValueError("noise_probability must be between 0 and 1.")

        # Copy the image
        img_array_noised = img_array.copy()
        height, width = img_array.shape[:2]
        is_color = len(img_array.shape) == 3  # True if RGB

        # Add noise
        for x in range(height):
            for y in range(width):
                if random.random() < noise_probability:
                    if random.random() < 0.5:
                        pixel = [255, 255, 255] if is_color else 255
                    else:
                        pixel = [0, 0, 0] if is_color else 0
                    img_array_noised[x, y] = pixel

        return img_array_noised

    except Exception as e:
        print(f"Error adding salt-and-pepper noise: {e}")
        return None
