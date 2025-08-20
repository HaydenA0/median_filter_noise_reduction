```python
def analyze_diff_growth(image_path, noise_probs):
    image_loader = ImageIO()
    original_img = image_loader.load_image(image_path, format="Grayscale")
    diffs = []

    # Compute diff for each noise probability
    for p in noise_probs:
        noisy_img = add_s_and_p_to_img(original_img, noise_probability=p)
        filtered_img = median_filter_bw(noisy_img)
        diff = image_loader.calculate_normal_difference_of_images(
            filtered_img, original_img
        )
        diffs.append(diff)

    diffs = np.array(diffs)

    # Compute numerical derivative d/dp diff
    d_diff = np.gradient(diffs, noise_probs)

    # Compute d'/d
    ratio = d_diff / diffs

    # Plot d'/d
    plt.figure(figsize=(8, 5))
    plt.plot(noise_probs, ratio, marker="o")
    plt.xlabel("Noise Probability")
    plt.ylabel("d'/d")
    plt.title("Analyzing Growth Rate: d'/d vs Noise Probability")
    plt.grid(True)
    plt.show()


noise_levels = np.linspace(0.01, 1, 45)  # avoid 0 to prevent division by zero
analyze_diff_growth("images/dog.png", noise_levels)
```

1. The Basic Observation: More Noise Creates Accelerating Error
The first graph showed a straightforward relationship: as the noise probability increases, the error ("Normalized Difference") in the filtered image also increases. Crucially, the curve is convex, meaning the error grows at an accelerating rate. The filter's performance degrades much more rapidly in high-noise environments.

2. The Deeper Insight: Identifying the Filter's "Breaking Point"
The second and third graphs, which plotted the proportional growth rate of the error (d'/d), provided the most critical insight. Instead of a simple curve, it revealed a distinct peak at a noise probability of roughly 0.3 to 0.4. This peak represents the filter's "breakdown threshold," allowing us to define three distinct operational zones:

    Safe Zone (Noise < 0.3): The filter is effective and reliably removes noise.

    Critical Zone (Noise ≈ 0.3 - 0.4): This is the tipping point where the filter's performance degrades most violently. It experiences a "cascading failure" as it starts mistaking noise for signal, leading to a compounding of errors.

    Failure Zone (Noise > 0.4): The filter is completely overwhelmed. The system is saturated with error, and applying the filter is more likely to harm the image than help it.

3. Explaining the System's Behavior at Extremes
We addressed two key questions about the filter's behavior:

    Why isn't the error 100% at 100% noise? At 100% noise, both the input and the filter's output are random collections of black and white pixels. Due to pure random chance, approximately 50% of the pixels in the filtered image will happen to match the original image, resulting in a maximum normalized difference of around 0.5, not 1.0.

    Why does the growth rate d'/d decrease after the peak? This is due to error saturation. After the peak, the image is already so corrupted (high d) that adding more noise has a diminishing marginal impact (lower d'). Since the denominator (d) is growing while the numerator (d') is shrinking, the ratio d'/d must decrease.

4. The Ultimate Conclusion: An Adaptive Filtering Algorithm
The most important takeaway from this analysis is a practical, intelligent algorithm for applying the filter:

    Estimate Noise: Before filtering, perform a quick analysis of the image to estimate the noise probability (e.g., by counting the percentage of pure black and white pixels).

    Compare to Threshold: Compare this estimated noise level to the empirically determined breakdown threshold (~0.3).

    Make an Informed Decision:

        If Noise < 0.3: The image is in the "Safe Zone." Proceed with applying the median filter.

        If Noise ≥ 0.3: The image is in the "Critical" or "Failure Zone." Do not apply the standard filter. Instead, choose a safer action, such as using a more robust filter, warning the user of severe corruption, or doing nothing to avoid making the image worse.
