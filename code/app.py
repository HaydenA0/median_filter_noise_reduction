# app.py
import streamlit as st
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import all the necessary functions from your existing modules
from image_io import ImageIO
from noise_generator import add_salt_and_pepper_noise as add_s_and_p_to_img
from median_filter import median_filter
from adaptive_median_filter import adaptive_median_filter

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Median Filter Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Caching ---
@st.cache_data
def load_image(image_path_or_buffer):
    """Loads an image and converts it to a grayscale numpy array."""
    image_loader = ImageIO()
    original_img = image_loader.load_image(image_path_or_buffer, format="Grayscale")
    return original_img


@st.cache_data
def analyze_filter_performance(filter_name, original_img, noise_probs, **kwargs):
    """
    Performs the d'/d analysis from your description.
    Accepts a filter name (string) which is hashable.
    """
    if filter_name == "Classic Median Filter":
        filter_func = median_filter
    else:
        filter_func = adaptive_median_filter

    image_loader = ImageIO()
    diffs = []

    progress_bar = st.progress(0, text="Analyzing performance...")
    total_steps = len(noise_probs)

    for i, p in enumerate(noise_probs):
        noisy_img = add_s_and_p_to_img(original_img, noise_probability=p)
        filtered_img = filter_func(noisy_img, **kwargs)
        diff = image_loader.calculate_normal_difference_of_images(
            filtered_img, original_img
        )
        diffs.append(diff)
        progress_bar.progress(
            (i + 1) / total_steps,
            text=f"Analyzing performance... (Noise Level {i+1}/{total_steps})",
        )

    progress_bar.empty()  # Clear the progress bar

    diffs = np.array(diffs)
    diffs[diffs == 0] = 1e-9

    d_diff = np.gradient(diffs, noise_probs)
    ratio = d_diff / diffs
    return diffs, d_diff, ratio


# --- Sidebar Controls ---
st.sidebar.title("🛠️ Controls")

# Image Selection
image_folder = "images/"
available_images = os.listdir(image_folder)
image_selection = st.sidebar.selectbox("Choose a sample image:", available_images)
uploaded_file = st.sidebar.file_uploader(
    "Or upload your own image:", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    original_image = load_image(uploaded_file)
else:
    original_image = load_image(os.path.join(image_folder, image_selection))

# Filter Selection
filter_type = st.sidebar.selectbox(
    "Select Filter Type:", ["Adaptive Median Filter", "Classic Median Filter"]
)

# Noise Control
noise_level = st.sidebar.slider(
    "Salt & Pepper Noise Probability (p)", 0.0, 1.0, 0.25, 0.01
)

# Generate Noisy Image (full resolution for visualization)
noisy_image = add_s_and_p_to_img(original_image, noise_probability=noise_level)

# Filter-specific parameters
st.sidebar.subheader(f"{filter_type} Parameters")
if filter_type == "Classic Median Filter":
    grid_size = st.sidebar.slider("Grid Size", 3, 15, 3, 2)
    max_size = grid_size
    filter_params = {"grid_size": grid_size}
    selected_filter_func = median_filter
else:  # Adaptive
    initial_size = st.sidebar.slider("Initial Grid Size", 3, 15, 3, 2)
    max_size = st.sidebar.slider("Max Grid Size", initial_size, 25, 7, 2)
    filter_params = {"initial_size": initial_size, "max_size": max_size}
    selected_filter_func = adaptive_median_filter

# --- Main Page ---
st.title("🔬 The Median Filter: An Interactive Deep Dive")
st.markdown(
    "This dashboard visualizes the behavior of median filters for removing salt-and-pepper noise. Use the controls on the left to change the image, noise level, and filter parameters."
)

# --- Section 1: The Filter's "Breaking Point" ---
st.header("1. The Filter's 'Breaking Point'")
st.markdown(
    "As described in the analysis, a filter's effectiveness isn't linear. There's a 'critical zone' where its performance degrades catastrophically. We can find this by plotting the **proportional growth rate of the error (`d'/d`)** against the noise probability. The peak of this curve is the filter's breaking point."
)

# ==================== NEW/MODIFIED SECTION START ====================
st.info(
    "💡 To speed things up, this analysis is run on a 128x128 thumbnail of the image."
)

# Create a button to trigger the long-running analysis
if st.button("📈 Run Performance Analysis", key="run_analysis"):

    # Create a small thumbnail for extremely fast analysis
    pil_img = Image.fromarray(original_image)
    thumbnail_size = (128, 128)
    # Use LANCZOS for high-quality downsampling
    resized_img = pil_img.resize(thumbnail_size, Image.Resampling.LANCZOS)
    analysis_image = np.array(resized_img)

    noise_levels_for_plot = np.linspace(
        0.01, 0.99, 40
    )  # Reduced points slightly for more speed

    # Call the analysis function on the SMALL image
    diffs, d_diff, ratio = analyze_filter_performance(
        filter_type, analysis_image, noise_levels_for_plot, **filter_params
    )

    # Store results in session state to persist them
    st.session_state["analysis_results"] = (noise_levels_for_plot, ratio)

# Check if results exist in session state and plot them
if "analysis_results" in st.session_state:
    noise_levels_for_plot, ratio = st.session_state["analysis_results"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(noise_levels_for_plot, ratio, marker=".", linestyle="-")
    ax.set_title(f"Proportional Error Growth Rate (d'/d) for {filter_type}")
    ax.set_xlabel("Noise Probability (p)")
    ax.set_ylabel("d'/d (Proportional Growth)")
    ax.grid(True)
    peak_noise = noise_levels_for_plot[np.argmax(ratio)]
    ax.axvline(
        x=noise_level, color="r", linestyle="--", label=f"Current p = {noise_level:.2f}"
    )
    ax.axvline(
        x=peak_noise,
        color="g",
        linestyle="--",
        label=f"Breaking Point ≈ {peak_noise:.2f}",
    )
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
# ===================== NEW/MODIFIED SECTION END =====================

# --- Section 2: Full Image Filtering ---
# (This section remains unchanged, using the full-resolution images)
st.header("2. Side-by-Side Comparison")

with st.spinner("Applying filter to the full image..."):
    filtered_image = selected_filter_func(noisy_image, **filter_params)

col1, col2 = st.columns(2)
with col1:
    st.image(
        noisy_image, caption=f"Noisy Image (p = {noise_level})", use_column_width=True
    )
with col2:
    st.image(
        filtered_image, caption=f"Filtered with {filter_type}", use_column_width=True
    )

# --- Section 3: Interactive Pixel-by-Pixel Explorer ---
# (This section also remains unchanged)
st.header("3. Interactive Pixel Explorer")
st.markdown("Select a pixel to see how the filter processes it step-by-step.")

rows, cols = noisy_image.shape
pad_width = max_size // 2

col_coord, row_coord = st.columns(2)
y = row_coord.slider("Select Pixel Row (Y-coordinate)", 0, rows - 1, rows // 2)
x = col_coord.slider("Select Pixel Column (X-coordinate)", 0, cols - 1, cols // 2)

# (The rest of the pixel explorer code is identical to the previous version)
# ...
vis_col1, vis_col2 = st.columns([1, 1])
with vis_col1:
    st.subheader("Image Context")
    fig, ax = plt.subplots()
    ax.imshow(noisy_image, cmap="gray")
    ax.set_title("Noisy Image")
    rect_current = patches.Rectangle(
        (x - 0.5, y - 0.5),
        1,
        1,
        linewidth=2,
        edgecolor="cyan",
        facecolor="none",
        label="Selected Pixel",
    )
    ax.add_patch(rect_current)
    ax.legend()
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with vis_col2:
    st.subheader("Algorithm Step-by-Step")
    padded_image = np.pad(noisy_image, pad_width, mode="symmetric")
    px, py = x + pad_width, y + pad_width

    if filter_type == "Classic Median Filter":
        k = grid_size // 2
        window = padded_image[py - k : py + k + 1, px - k : px + k + 1]
        z_med = np.median(window)
        output_pixel = z_med

        st.info(f"**Classic Filter Logic (Size: {grid_size}x{grid_size})**")
        st.write("1. Extract the window.")
        st.write(f"2. Calculate the median = **{z_med:.0f}**")
        st.success(f"**Final Output: {output_pixel:.0f}**")

    else:  # Adaptive Filter Logic
        st.info(
            f"**Adaptive Filter Logic (Initial: {initial_size}x{initial_size}, Max: {max_size}x{max_size})**"
        )
        window_size = initial_size
        output_pixel = None

        while window_size <= max_size:
            st.markdown(f"--- \n**Stage: Window Size = {window_size}x{window_size}**")
            k = window_size // 2
            window = padded_image[py - k : py + k + 1, px - k : px + k + 1]

            z_min, z_max, z_med = np.min(window), np.max(window), np.median(window)
            z_xy = noisy_image[y, x]

            st.code(f"z_min = {z_min}, z_max = {z_max}, z_med = {z_med}, z_xy = {z_xy}")

            if z_min < z_med < z_max:
                st.write(f"-> `z_med` is not noise. Checking `z_xy`...")
                if z_min < z_xy < z_max:
                    st.write(f"-> `z_xy` is not noise. Keeping original.")
                    output_pixel = z_xy
                    st.success(f"**Final Output: {output_pixel}**")
                    break
                else:
                    st.write(f"-> `z_xy` is noise. Using median.")
                    output_pixel = z_med
                    st.success(f"**Final Output: {output_pixel:.0f}**")
                    break
            else:
                st.write(f"-> `z_med` could be noise. Increasing window size.")
                window_size += 2
                if window_size > max_size:
                    st.warning(f"Max size reached. Using `z_med`.")
                    output_pixel = z_med
                    st.success(f"**Final Output: {output_pixel:.0f}**")
                    break

    fig, ax = plt.subplots()
    ax.imshow(window, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            ax.text(j, i, int(window[i, j]), ha="center", va="center", color="red")
    ax.set_title(f"Zoomed Window ({window.shape[0]}x{window.shape[0]})")
    ax.set_xticks([]), ax.set_yticks([])
    st.pyplot(fig)
    plt.close(fig)
