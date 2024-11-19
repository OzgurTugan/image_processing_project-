# Image Processing Filters

A Python-based image processing tool that applies customizable filters with Z-score normalization and visualizes results with histograms.

## Features
- Z-Score Normalization applied to images before filter processing.
- Supports various filters like Gaussian Blur, Median Blur, Bilateral Filter, Laplacian, and more.
- Visualizes histograms of the original, normalized, and filtered images for comparison.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processing-filters.git   
2. Navigate to the project directory:
    ```bash
   cd image-processing-filters
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
## Usage
1. Import the function in your Python script:
   ```bash
   from image_processing import apply_filters_to_image
2. Run the function with default parameters:
   ```bash
   apply_filters_to_image("path_to_your_image.jpg")
3. Customize filter parameters:
   ```bash
   apply_filters_to_image(
    "path_to_your_image.jpg",
    gaussian_params=(3, 3, 0.5),
    median_ksize=7,
    bilateral_params=(15, 50, 100),
    gabor_params=(31, 3.0, np.pi / 4, 8.0, 0.8),
    fourier_mask_size=50,
   )
   
## Example
1. An example image is provided in the example_images/ directory. To test the function, run:
   ```bash
   apply_filters_to_image("example_images/example_image.jpg")
