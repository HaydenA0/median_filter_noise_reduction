from image_io import *
from noise_generator import *
from median_filter import *
from adaptive_median_filter import *


image_loader = ImageIO()
img_array = add_salt_and_pepper_noise(
    image_loader.load_image("images/flask.png", format="Grayscale"),
    noise_probability=0.4,
)
adaptive = adaptive_median_filter_vec_compiled(img_array, initial_size=3, max_size=19)
image_loader.compare_images([img_array, adaptive], ["Original", "Adaptive"])

# filtered_img = median_filter(img_array)
