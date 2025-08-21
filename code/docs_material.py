import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Import your project's custom modules ---
from image_io import ImageIO
from noise_generator import add_salt_and_pepper_noise
from median_filter import median_filter
from adaptive_median_filter_p import adaptive_median_filter
from adaptive_median_filter_p_compiled import (
    adaptive_median_filter_wrapper,
)  # 'Just Compiled' version
from adaptive_median_filter import (
    adaptive_median_filter_vec_compiled,
)  # 'Vectorized + Compiled'

# --- Configuration ---
OUTPUT_DIR = "docs/images/"
SAMPLE_IMAGE_PATH = "images/woman.png"  # A good detailed image for examples
NOISE_LEVEL_DEMO = 0.4  # A challenging noise level for visual comparison
ADAPTIVE_PARAMS = {"initial_size": 3, "max_size": 19}
CLASSIC_PARAMS = {"grid_size": 3}


def setup_output_directory():
    """Ensures the output directory for generated images exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' is ready.")


def generate_intro_images(image_loader):
    """
    Creates a 4-panel image showing the problem and the solutions.
    (Original -> Noisy -> Classic Filter -> Adaptive Filter)
    """
    print("Generating introduction comparison image...")
    original_img = image_loader.load_image(SAMPLE_IMAGE_PATH, format="Grayscale")
    noisy_img = add_salt_and_pepper_noise(
        original_img, noise_probability=NOISE_LEVEL_DEMO
    )
    classic_filtered = median_filter(noisy_img, **CLASSIC_PARAMS)
    adaptive_filtered = adaptive_median_filter_vec_compiled(
        noisy_img, **ADAPTIVE_PARAMS
    )

    images = [original_img, noisy_img, classic_filtered, adaptive_filtered]
    titles = [
        "1. Original Image",
        f"2. With {NOISE_LEVEL_DEMO*100}% Noise",
        "3. Classic Median Filter",
        "4. Adaptive Median Filter",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "filter_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison image to '{output_path}'")


def generate_performance_plot():
    """
    Creates a bar chart comparing the performance of different filter implementations.
    Data is hardcoded from the speed_comparison.py logs for consistency.
    """
    print("Generating performance comparison plot...")

    # Data from your logs for a large image (e.g., bear.png @ 881x874)
    labels = [
        "Pure Python\n(Classic)",
        "Pure Python\n(Adaptive)",
        "Numba JIT Compiled\n(Loops)",
        "Vectorized + Numba\n(Fastest)",
    ]
    times = [2.0297, 12.1415, 0.5856, 0.3534]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, times, color=["#ff9999", "#ff6666", "#99ff99", "#66b3ff"])

    ax.set_yscale("log")
    ax.set_ylabel("Execution Time (seconds, log scale)")
    ax.set_title(
        "Filter Performance Comparison on a Large Image (bear.png)", fontsize=16
    )
    ax.bar_label(bars, fmt="%.4f s", padding=3)

    # Add a horizontal line for the 1-second mark for context
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.text(
        len(labels) - 0.5, 1.05, "1-second mark", color="gray", va="bottom", ha="right"
    )

    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "performance_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved performance plot to '{output_path}'")


def run_error_analysis(image_loader, filter_func, filter_params, noise_probs):
    """
    Helper to run the error analysis for a given filter.
    Uses a thumbnail for speed.
    """
    pil_img = Image.open(SAMPLE_IMAGE_PATH).convert("L")
    thumbnail_size = (128, 128)
    original_img = np.array(pil_img.resize(thumbnail_size, Image.Resampling.LANCZOS))

    diffs = []
    for p in noise_probs:
        noisy_img = add_salt_and_pepper_noise(original_img, noise_probability=p)
        filtered_img = filter_func(noisy_img, **filter_params)
        diff = image_loader.calculate_normal_difference_of_images(
            filtered_img, original_img
        )
        diffs.append(diff)

    diffs = np.array(diffs)
    diffs[diffs == 0] = 1e-9  # Avoid division by zero

    d_diff = np.gradient(diffs, noise_probs)
    ratio = d_diff / diffs

    return noise_probs, diffs, ratio


def generate_breaking_point_analysis(image_loader):
    """
    Generates the plots that visualize the filter's "breaking point".
    """
    print("Generating 'breaking point' analysis plots...")
    noise_levels = np.linspace(0.01, 0.99, 50)

    # We will analyze the classic filter as its behavior is simpler to interpret
    # for this demonstration.
    noise, diffs, ratio = run_error_analysis(
        image_loader, median_filter, CLASSIC_PARAMS, noise_levels
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # --- Plot 1: Error vs. Noise ---
    ax1.plot(noise, diffs, marker=".", linestyle="-", color="dodgerblue")
    ax1.set_title("Observation: Error Accelerates with More Noise", fontsize=16)
    ax1.set_xlabel("Noise Probability (p)")
    ax1.set_ylabel("Normalized Error (d)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Plot 2: Proportional Growth Rate (d'/d) ---
    ax2.plot(noise, ratio, marker=".", linestyle="-", color="crimson")
    ax2.set_title("Insight: Identifying the Filter's 'Breaking Point'", fontsize=16)
    ax2.set_xlabel("Noise Probability (p)")
    ax2.set_ylabel("Proportional Error Growth (d'/d)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Find the peak and annotate the zones
    peak_index = np.argmax(ratio)
    peak_noise = noise[peak_index]

    # Shade the zones for clarity
    ax2.axvspan(0, peak_noise * 0.8, color="green", alpha=0.15, label="Safe Zone")
    ax2.axvspan(
        peak_noise * 0.8,
        peak_noise * 1.2,
        color="orange",
        alpha=0.2,
        label="Critical Zone",
    )
    ax2.axvspan(peak_noise * 1.2, 1.0, color="red", alpha=0.15, label="Failure Zone")

    # Add a line for the peak
    ax2.axvline(
        x=peak_noise,
        color="black",
        linestyle="--",
        label=f"Breaking Point ≈ {peak_noise:.2f}",
    )
    ax2.legend()

    plt.tight_layout(pad=3.0)
    output_path = os.path.join(OUTPUT_DIR, "breaking_point_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved breaking point analysis to '{output_path}'")


if __name__ == "__main__":
    # Initialize necessary components
    image_loader = ImageIO()

    # 1. Create the output directory
    setup_output_directory()

    # 2. Generate the main comparison image
    generate_intro_images(image_loader)

    # 3. Generate the performance bar chart
    generate_performance_plot()

    # 4. Generate the detailed d'/d analysis plots
    generate_breaking_point_analysis(image_loader)

    print("\n✅ All documentation materials have been successfully generated.")
