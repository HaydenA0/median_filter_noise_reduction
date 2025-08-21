import os
import timeit
import cProfile
import pstats
import numpy as np
from memory_profiler import memory_usage

# --- Assuming these are your custom modules ---
# Make sure they are in the same directory or Python path
from image_io import *
from noise_generator import *
from median_filter import *
from adaptive_median_filter import *

# --- Configuration ---
IMAGE_DIR = "images"
IMAGE_FILES = [
    "flask.png",  # ~83k
    "model.png",  # ~49k
    "dog.png",  # ~302k
    "woman.png",  # ~684k
    "bear.png",  # ~1.2M
]
NOISE_PROBABILITY = 0.4
ADAPTIVE_MAX_SIZE = 19
TIMING_RUNS = 5  # Number of times to run each filter for timing statistics

# --- Main Test Harness ---


def run_performance_analysis():
    """
    Main function to run the full performance analysis suite.
    """
    # To store summary results for the final table
    summary_results = []

    print("=========================================================")
    print("=          Median Filter Performance Analysis           =")
    print("=========================================================\n")

    # Initialize the image loader once
    image_loader = ImageIO()

    for image_file in IMAGE_FILES:
        image_path = os.path.join(IMAGE_DIR, image_file)
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found at {image_path}. Skipping.")
            continue

        print(f"---------------------------------------------------------")
        print(f"Processing Image: {image_file}")
        print(f"---------------------------------------------------------\n")

        # 1. Load image and add noise (do this once per image)
        try:
            original_img = image_loader.load_image(image_path, format="Grayscale")
            noisy_img = add_salt_and_pepper_noise(
                original_img, noise_probability=NOISE_PROBABILITY
            )
            print(f"Image '{image_file}' loaded. Dimensions: {original_img.shape}\n")
        except Exception as e:
            print(f"ERROR: Could not load or process {image_file}. Reason: {e}")
            continue

        # Dictionary to hold results for this image
        current_image_results = {"image": image_file, "dims": original_img.shape}

        # --- Test Functions ---
        # We define them here to capture the 'noisy_img' variable
        def run_adaptive():
            adaptive_median_filter_vec_compiled(noisy_img, max_size=ADAPTIVE_MAX_SIZE)

        def run_classic():
            median_filter(noisy_img)

        # --- Analysis Loop for both filters ---
        filters_to_test = {
            "Adaptive Median Filter": run_adaptive,
            "Classic Median Filter": run_classic,
        }

        for name, func in filters_to_test.items():
            print(f"--- Analyzing: {name} ---")

            # A. Execution Time (using timeit for statistics)
            print("1. Execution Time Analysis...")
            try:
                times = timeit.repeat(func, number=1, repeat=TIMING_RUNS)
                mean_time = np.mean(times)
                std_dev = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)

                print(f"   Runs: {TIMING_RUNS}")
                print(f"   - Mean Time: {mean_time:.4f} seconds")
                print(f"   - Std Dev:   {std_dev:.4f} seconds")
                print(f"   - Min Time:  {min_time:.4f} seconds")
                print(f"   - Max Time:  {max_time:.4f} seconds\n")
                current_image_results[f"{name}_time"] = mean_time
            except Exception as e:
                print(f"   ERROR during timing: {e}\n")
                current_image_results[f"{name}_time"] = -1

            # B. Memory Usage (using memory_profiler)
            print("2. Memory Usage Analysis...")
            try:
                # memory_usage returns a list of memory samples during function execution
                mem_usage = memory_usage((func,), interval=0.1, max_usage=True)
                # max_usage=True returns a single float of the peak memory
                peak_mem = mem_usage if isinstance(mem_usage, float) else max(mem_usage)

                print(f"   - Peak Memory Usage: {peak_mem:.2f} MiB\n")
                current_image_results[f"{name}_mem"] = peak_mem
            except Exception as e:
                print(f"   ERROR during memory profiling: {e}\n")
                current_image_results[f"{name}_mem"] = -1

            # C. CPU Profiling (using cProfile)
            print("3. CPU Hotspot Analysis (Top 10 functions by cumulative time)...")
            try:
                profiler = cProfile.Profile()
                profiler.enable()
                func()  # Run the function once under the profiler
                profiler.disable()

                # Create a Stats object and print the most time-consuming parts
                stats = pstats.Stats(profiler).sort_stats("cumulative")
                stats.print_stats(10)  # Print top 10 hotspots
                print("\n")
            except Exception as e:
                print(f"   ERROR during CPU profiling: {e}\n")

        summary_results.append(current_image_results)
        print(f"Finished analysis for {image_file}\n")

    # --- Print Final Summary Table ---
    print_summary(summary_results)


def print_summary(results):
    """
    Prints a formatted summary table of all results.
    """
    print(
        "========================================================================================="
    )
    print(
        "=                                 Overall Performance Summary                           ="
    )
    print(
        "========================================================================================="
    )
    print(
        f"{'Image':<15} | {'Dimensions':<15} | {'Adaptive Time (s)':<20} | {'Classic Time (s)':<20} | {'Winner'}"
    )
    print("-" * 90)

    for res in results:
        image = res["image"]
        dims = str(res["dims"])
        adaptive_time = res.get("Adaptive Median Filter_time", "N/A")
        classic_time = res.get("Classic Median Filter_time", "N/A")

        if (
            adaptive_time != "N/A"
            and classic_time != "N/A"
            and adaptive_time > 0
            and classic_time > 0
        ):
            adaptive_time_f = f"{adaptive_time:.4f}"
            classic_time_f = f"{classic_time:.4f}"
            winner = "Classic" if classic_time < adaptive_time else "Adaptive"
        else:
            adaptive_time_f = "FAIL"
            classic_time_f = "FAIL"
            winner = "N/A"

        print(
            f"{image:<15} | {dims:<15} | {adaptive_time_f:<20} | {classic_time_f:<20} | {winner}"
        )
    print(
        "=========================================================================================\n"
    )


if __name__ == "__main__":
    run_performance_analysis()
