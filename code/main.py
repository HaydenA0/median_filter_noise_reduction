from image_io import *
from noise_generator import *
from median_filter import *
from adaptive_median_filter import *

N = 10


image_loader = ImageIO()
img_array = add_salt_and_pepper_noise(
    image_loader.load_image("images/flask.png", format="Grayscale"),
    noise_probability=0.85,
)
clear_img = adaptive_median_filter(img_array, max_size=19)
image_loader.compare_images([img_array, clear_img], ["Original", "Cleared"])

# filtered_img = median_filter(img_array)
