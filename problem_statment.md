### 1. The Pinned Location

> A median filter is an example of a nonlinear filter and, if properly designed, is very good at preserving image detail. To run a median filter:
>
> *   consider each pixel in the image
> *   sort the neighbouring pixels into order based upon their intensities
> *   replace the original value of the pixel with the median value from the list
>
> A median filter is a rank-selection (RS) filter... Median and other RCRS filters are good at removing salt and pepper noise from an image, and also cause relatively little blurring of edges, and hence are often used in computer vision applications.

### 2. The Justification

This is an ideal candidate for a **Low-Level Programming / Graphics Simulation** challenge.

*   **Explicit Algorithm:** The text provides a clear, step-by-step description of the algorithm, making it perfect for implementation without needing external domain knowledge.
*   **Fundamental Concept:** The median filter is a classic and fundamental algorithm in image processing. Implementing it from scratch provides valuable insight into how filters work at a low level, rather than just calling a library function.
*   **Visual Feedback:** The result of the algorithm is visual. You can generate an image with "salt and pepper noise" (also described in the article) and visually confirm that your filter is working correctly, which is highly rewarding.
*   **Scalable Complexity:** The core concept is simple, but handling edge cases, color channels, and performance optimization provides significant depth for more advanced challenges.

### 3. The Programming Problem

Your challenge is to implement a median filter to remove "salt and pepper" noise from an image.

You will need a way to represent an image (a 2D array for grayscale, or a 3D array for color) and a function to add salt and pepper noise (randomly setting some pixels to black or white).

---

#### **Level 1: The Basic Grayscale Filter**

**Objective:** Implement a simple 3x3 median filter for a grayscale image.

1.  **Input:** A 2D array of integers representing a grayscale image (e.g., values from 0 to 255).
2.  **Noise Generation:** Create a function that takes an image and adds salt and pepper noise. It should iterate through the pixels and, with a small probability (e.g., 5%), set a pixel to either 0 (black) or 255 (white).
3.  **Filter Implementation:**
    *   Write a function `median_filter(image)` that takes the noisy image as input.
    *   Create a new 2D array for the output image to avoid modifying the image while you're still reading from it.
    *   Iterate over every pixel `(x, y)` in the input image, but **for now, you can ignore the border pixels** to keep it simple.
    *   For each pixel, collect the intensity values of the 3x3 grid of its neighbors (including the pixel itself).
    *   Sort this list of 9 values.
    *   Find the median value (the 5th element in the sorted list).
    *   Set the value of the pixel `(x, y)` in your output image to this median value.
4.  **Output:** The denoised 2D array. You can test your function by printing a small noisy array before and after filtering. If you can display images, compare the noisy input with the clean output.

---

#### **Level 2: Color Images and Edge Handling**

**Objective:** Extend your filter to work with color images and properly handle the image borders.

1.  **Edge Handling:** Modify your Level 1 filter to process the pixels at the edges and corners of the image. Common strategies include:
    *   **Padding:** Add a border of pixels around the image (e.g., by replicating the edge pixels or using black pixels) so the 3x3 window is always full.
    *   **Shrinking Window:** Use a smaller window for edge/corner pixels (e.g., a 2x3 or 2x2 window). This is more complex to implement.
    *   **Clamping:** When the window goes outside the image boundary, clamp the coordinates to the nearest valid pixel.
2.  **Color Filtering:** Adapt your algorithm to work on an RGB color image (represented as a 3D array of `[height][width][3]`).
    *   Apply the median filter to each color channel (Red, Green, Blue) independently.
    *   For a pixel `(x, y)`, you will collect the Red values from its neighbors, find the median, and use that for the new Red value. Repeat this process for the Green and Blue channels.

---

#### **Level 3: Optimization and Generalization**

**Objective:** Make your filter more efficient and flexible.

1.  **Variable Window Size:** Generalize your filter to accept an odd window size `N` (e.g., 5 for a 5x5 window, 7 for a 7x7 window, etc.) as a parameter.
2.  **Performance Optimization:** The naive approach of collecting and sorting the list of neighbors for every single pixel is computationally expensive, especially for large images or large window sizes.
    *   **Challenge:** Implement a faster median filter. A common advanced technique involves using a moving histogram. As you slide the window one pixel to the right, you don't need to rebuild the entire histogram; you can just subtract the values from the column that left the window and add the values from the new column that entered. This is a much more complex data structure challenge but results in a significant performance boost.
3.  **Comparison:** Implement a simple "mean filter" (or "box blur") where you take the average of the neighbors instead of the median. Apply both the mean and median filters to the same noisy image and compare the results. You should see that the median filter is much better at preserving sharp edges, as the article claims.

Sources :
https://en.wikipedia.org/wiki/Noise_reduction
https://en.wikipedia.org/wiki/Median_filter
