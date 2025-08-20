import streamlit as st
import numpy as np
from PIL import Image
import os

# Import your existing, well-structured modules
from image_io import ImageIO
from noise_generator import add_salt_and_pepper_noise
from median_filter import median_filter
from adaptive_median_filter import adaptive_median_filter

# Initialize your helper class
image_loader = ImageIO()

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Interactive Image Filter Explorer")

st.title("🔬 Interactive Image Filter Explorer")
st.markdown(
    """
This app visualizes the effect of Median and Adaptive Median filters on images with salt-and-pepper noise. 
It's designed to explore the "breaking point" of these filters as described in the analysis.

**How to use:**
1.  **Choose an image** from the sidebar (or upload your own).
2.  **Adjust the noise level** to see how corruption affects the image.
3.  **Select a filter** and configure its parameters.
4.  **Click "Apply Noise & Filter"** to see the results.
"""
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

# 1. Image Selection
image_folder = "images/"
available_images = [
    f for f in os.listdir(image_folder) if f.endswith(("png", "jpg", "jpeg"))
]

uploaded_file = st.sidebar.file_uploader(
    "Upload your own image", type=["png", "jpg", "jpeg"]
)
if uploaded_file is None:
    selected_image_name = st.sidebar.selectbox(
        "Or choose an example image:", available_images
    )
    image_path = os.path.join(image_folder, selected_image_name)
    original_img = image_loader.load_image(image_path, format="Grayscale")
else:
    original_img = np.array(Image.open(uploaded_file).convert("L"))

# 2. Noise Control
noise_prob = st.sidebar.slider("Salt & Pepper Noise Probability", 0.0, 1.0, 0.25, 0.01)

# 3. Filter Selection
filter_type = st.sidebar.selectbox(
    "Choose Filter", ("None", "Standard Median Filter", "Adaptive Median Filter")
)

# 4. Filter-specific Parameters
if filter_type == "Standard Median Filter":
    st.sidebar.subheader("Standard Filter Settings")
    kernel_size = st.sidebar.slider("Kernel Size (odd numbers)", 3, 21, 3, 2)

elif filter_type == "Adaptive Median Filter":
    st.sidebar.subheader("Adaptive Filter Settings")
    initial_size = st.sidebar.slider("Initial Window Size", 3, 19, 3, 2)
    max_size = st.sidebar.slider("Max Window Size", initial_size + 2, 25, 7, 2)
    if max_size < initial_size:
        st.sidebar.warning("Max Size should be greater than or equal to Initial Size.")


# --- Main Application Logic ---

# Use session state to store images and prevent re-computation on every interaction
if "noisy_img" not in st.session_state:
    st.session_state.noisy_img = None
if "filtered_img" not in st.session_state:
    st.session_state.filtered_img = None
if "diff_score" not in st.session_state:
    st.session_state.diff_score = 0.0


if st.sidebar.button("🚀 Apply Noise & Filter", use_container_width=True):
    with st.spinner("Processing..."):
        # Step 1: Add noise
        noisy_image = add_salt_and_pepper_noise(
            original_img, noise_probability=noise_prob
        )
        st.session_state.noisy_img = noisy_image

        # Step 2: Apply the selected filter
        if filter_type == "Standard Median Filter":
            filtered_image = median_filter(noisy_image, grid_size=kernel_size)
        elif filter_type == "Adaptive Median Filter":
            filtered_image = adaptive_median_filter(
                noisy_image, initial_size=initial_size, max_size=max_size
            )
        else:  # "None" filter
            filtered_image = noisy_image

        st.session_state.filtered_img = filtered_image

        # Step 3: Calculate difference
        diff = image_loader.calculate_normal_difference_of_images(
            filtered_image, original_img
        )
        st.session_state.diff_score = diff if diff is not None else 0.0


# --- Display Results ---
st.header("🖼️ Image Comparison")

# Define columns for a side-by-side view
col1, col2, col3 = st.columns(3)

with col1:
    st.image(original_img, caption="1. Original Image", use_column_width=True)

with col2:
    if st.session_state.noisy_img is not None:
        st.image(
            st.session_state.noisy_img,
            caption=f"2. Noisy Image (p={noise_prob:.2f})",
            use_column_width=True,
        )
    else:
        st.info("Apply noise and filter to see the noisy image here.")

with col3:
    if st.session_state.filtered_img is not None:
        st.image(
            st.session_state.filtered_img,
            caption=f"3. Filtered Image ({filter_type})",
            use_column_width=True,
        )
    else:
        st.info("Apply noise and filter to see the result here.")


# --- Performance Metric and Analysis ---
st.header("📊 Performance Analysis")

if st.session_state.diff_score > 0:
    # Use a metric card to display the error
    st.metric(
        label="Normalized Difference (Error)",
        value=f"{st.session_state.diff_score:.4f}",
    )

    # Provide contextual feedback based on the analysis
    if noise_prob < 0.3:
        st.success(
            f"**Safe Zone (Noise < 0.3):** The noise level is low. The filter should be effective."
        )
    elif 0.3 <= noise_prob <= 0.45:
        st.warning(
            f"**Critical Zone (Noise ≈ 0.3 - 0.45):** The filter is at its breaking point. Performance will degrade rapidly here."
        )
    else:
        st.error(
            f"**Failure Zone (Noise > 0.45):** The filter is overwhelmed. The output may be worse than the noisy image."
        )
else:
    st.info("Click the 'Apply' button to calculate the performance metric.")
