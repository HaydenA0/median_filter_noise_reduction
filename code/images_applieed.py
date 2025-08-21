import os
import numpy as np
import matplotlib.pyplot as plt

# --- Import your project's custom modules ---
from image_io import ImageIO
from noise_generator import add_salt_and_pepper_noise
from median_filter import median_filter

# Use the fastest, most optimized adaptive filter for the final comparison
from adaptive_median_filter import *

# --- Configuration ---
OUTPUT_DIR = "docs/images/"
IMAGE_DIR = "images/"
# Define the representative noise levels to showcase the filters' behavior
NOISE_LEVELS_GALLERY = [0.10, 0.40, 0.70]  # Low, Critical, and High noise

# Use the same parameters as before for consistency
ADAPTIVE_PARAMS = {"initial_size": 3, "max_size": 19}
CLASSIC_PARAMS = {"grid_size": 3}


def setup_output_directory():
    """Ensures the output directory for generated images exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' is ready.")


def generate_filter_gallery(image_loader, image_path):
    """
    Creates a grid of images for a single input file, showing the performance
    of both filters across multiple noise levels.
    """
    base_name = os.path.basename(image_path)
    print(f"Generating filter gallery for '{base_name}'...")

    original_img = image_loader.load_image(image_path, format="Grayscale")

    # Set up the plot grid: rows for noise levels, columns for image types
    num_rows = len(NOISE_LEVELS_GALLERY)
    num_cols = 3  # Columns: Noisy Input, Classic Filter, Adaptive Filter
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten axes array for easier iteration if there's only one row
    if num_rows == 1:
        axes = np.array([axes])

    for i, p in enumerate(NOISE_LEVELS_GALLERY):
        # Generate noisy image and apply filters
        noisy_img = add_salt_and_pepper_noise(original_img, noise_probability=p)
        classic_filtered = median_filter(noisy_img, **CLASSIC_PARAMS)
        adaptive_filtered = adaptive_median_filter_vec_compiled(
            noisy_img, **ADAPTIVE_PARAMS
        )

        images_in_row = [noisy_img, classic_filtered, adaptive_filtered]

        for j, img in enumerate(images_in_row):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            # Set column titles only for the top row
            if i == 0:
                if j == 0:
                    ax.set_title("1. Noisy Input", fontsize=14, pad=10)
                elif j == 1:
                    ax.set_title("2. Classic Filter Output", fontsize=14, pad=10)
                else:
                    ax.set_title("3. Adaptive Filter Output", fontsize=14, pad=10)

        # Set a clear label for each row on the left-most plot
        axes[i, 0].set_ylabel(f"Noise = {p*100:.0f}%", fontsize=16, labelpad=20)

    # Add a main title for the entire figure
    fig.suptitle(f"Filter Comparison Gallery: {base_name}", fontsize=20, y=0.97)
    fig.subplots_adjust(top=0.92)  # Adjust layout to make space for suptitle

    # Save the composite image
    output_filename = f"gallery_results_{os.path.splitext(base_name)[0]}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Saved gallery image to '{output_path}'")


if __name__ == "__main__":
    # Initialize necessary components
    image_loader = ImageIO()

    # 1. Create the output directory
    setup_output_directory()

    # 2. Find all .png images in the image directory
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png")]

    if not image_files:
        print("No PNG images found in the 'images/' directory. Nothing to do.")
    else:
        # 3. Generate a gallery for each image
        for image_file in image_files:
            full_path = os.path.join(IMAGE_DIR, image_file)
            generate_filter_gallery(image_loader, full_path)

        print(
            "\n✅ All extended documentation galleries have been successfully generated."
        )
