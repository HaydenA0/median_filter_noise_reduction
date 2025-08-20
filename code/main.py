from image_io import *
from noise_generator import *
from median_filter import *

image_loader = ImageIO()
img_array = image_loader.load_image("images/woman.png", format="RGB")
clear_img = median_filter(img_array, grid_size=5)
image_loader.compare_images([img_array, clear_img], ["Original", "Cleared"])

# filtered_img = median_filter(img_array)
