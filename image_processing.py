import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load an image, invert its colors, and normalize it.
def load_image(image_path):
    """
    Loads the image, inverts its colors, and returns it.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Inverted grayscale image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image could not be loaded. Please check the file path.")
    
    # Invert the image colors (white to black and vice versa)
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

# Function to apply Z-Score normalization to an image.
def z_score_normalisation(image):
    """
    Normalizes the image using Z-Score normalization.
    
    Args:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Z-Score normalized image.
    """
    mean, std = np.mean(image), np.std(image)
    normalized = (image - mean) / std
    normalized = ((normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))) * 255
    return normalized.astype(np.uint8)

# Function to apply a specified filter to an image with optional parameters.
def apply_filter(image, filter_name, **kwargs):
    """
    Applies a specified filter to the image with customizable parameters.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        filter_name (str): Name of the filter to apply.
        **kwargs: Additional parameters specific to the filter.

    Returns:
        numpy.ndarray: Filtered image.
    """
    if filter_name == "GaussianBlur":
        return cv2.GaussianBlur(image, kwargs.get("ksize", (5, 5)), kwargs.get("sigma", 1))
    elif filter_name == "MedianBlur":
        return cv2.medianBlur(image, kwargs.get("ksize", 5))
    elif filter_name == "BilateralFilter":
        return cv2.bilateralFilter(image, kwargs.get("d", 9), kwargs.get("sigma_color", 75), kwargs.get("sigma_space", 75))
    elif filter_name == "LaplacianFilter":
        laplacian = cv2.Laplacian(image, kwargs.get("ddepth", cv2.CV_64F))
        return cv2.convertScaleAbs(laplacian)
    elif filter_name == "SobelFilter":
        return cv2.Sobel(image, cv2.CV_64F, kwargs.get("dx", 1), kwargs.get("dy", 0), kwargs.get("ksize", 3))
    elif filter_name == "CannyEdge":
        return cv2.Canny(image, kwargs.get("threshold1", 100), kwargs.get("threshold2", 200))
    elif filter_name == "AdaptiveThreshold":
        return cv2.adaptiveThreshold(
            image, kwargs.get("max_value", 255), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            kwargs.get("block_size", 11), kwargs.get("c", 2)
        )
    elif filter_name == "TopHat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kwargs.get("kernel_size", 5),) * 2)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif filter_name == "BlackHat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kwargs.get("kernel_size", 5),) * 2)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    elif filter_name == "GaborFilter":
        gabor_kernel = cv2.getGaborKernel(
            (kwargs.get("ksize", 31), kwargs.get("ksize", 31)), kwargs.get("sigma", 4.0), 
            kwargs.get("theta", 0), kwargs.get("lambd", 10.0), kwargs.get("gamma", 0.5), 0, ktype=cv2.CV_32F
        )
        return cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    elif filter_name == "FourierFilter":
        mask_size = kwargs.get("mask_size", 30)
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - mask_size:crow + mask_size, ccol - mask_size:ccol + mask_size] = 1
        filtered_dft = dft_shift * mask
        dft_ishift = np.fft.ifftshift(filtered_dft)
        img_back = cv2.idft(dft_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    elif filter_name == "MinFilter":
        kernel_size = kwargs.get("kernel_size", 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel)
    elif filter_name == "MaxFilter":
        kernel_size = kwargs.get("kernel_size", 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel)
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

# Function to calculate the histogram of an image.
def calculate_histogram(image):
    """
    Calculates the histogram of the image.
    
    Args:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Histogram data.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

# Function to visualize the results using Matplotlib.
def visualize_results(results):
    """
    Visualizes the original, normalized, and filtered images along with histograms.
    
    Args:
        results (list): List of tuples containing filter name, original image, normalized image, 
                        filtered image, and histograms.
    """
    for idx, (filter_name, original, normalized, filtered, histograms) in enumerate(results):
        plt.figure(figsize=(18, 6))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(original, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Original Image\n{filter_name}")
        plt.axis("off")
        
        # Normalized image
        plt.subplot(1, 4, 2)
        plt.imshow(normalized, cmap="gray", vmin=0, vmax=255)
        plt.title("Z-Score Normalized")
        plt.axis("off")
        
        # Filtered image
        plt.subplot(1, 4, 3)
        plt.imshow(filtered, cmap="gray", vmin=0, vmax=255)
        plt.title("Filtered Image")
        plt.axis("off")
        
        # Histogram comparison
        plt.subplot(1, 4, 4)
        plt.plot(histograms["Original"], label="Original", color="blue")
        plt.plot(histograms["Normalized"], label="Normalized", color="green")
        plt.plot(histograms["Filtered"], label="Filtered", color="red")
        plt.title("Histogram Comparison")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Default process function with fixed parameters.
def process_image(image_path):
    """
    Processes the image using default parameters.
    
    Args:
        image_path (str): Path to the image file.
    """
    process_image_with_parameters(image_path)

# Function to process the image with customizable filter parameters.
def process_image_with_parameters(
    image_path,
    gaussian_params=(5, 5, 1),
    median_ksize=5,
    bilateral_params=(9, 75, 75),
    laplacian_ddepth=cv2.CV_64F,
    sobel_params=(1, 0, 3),
    canny_params=(100, 200),
    adaptive_threshold_params=(255, 11, 2),
    top_hat_kernel=5,
    black_hat_kernel=5,
    gabor_params=(31, 4.0, 0, 10.0, 0.5),
    fourier_mask_size=30,
    min_filter_kernel=3,
    max_filter_kernel=3,
):
    """
    Processes the image using customizable filter parameters and visualizes the results.
    
    Args:
        image_path (str): Path to the image file.
        gaussian_params (tuple): Parameters for GaussianBlur.
        median_ksize (int): Kernel size for MedianBlur.
        bilateral_params (tuple): Parameters for BilateralFilter.
        laplacian_ddepth (int): Depth for LaplacianFilter.
        sobel_params (tuple): Parameters for SobelFilter.
        canny_params (tuple): Parameters for CannyEdge detection.
        adaptive_threshold_params (tuple): Parameters for AdaptiveThreshold.
        top_hat_kernel (int): Kernel size for TopHat transformation.
        black_hat_kernel (int): Kernel size for BlackHat transformation.
        gabor_params (tuple): Parameters for GaborFilter.
        fourier_mask_size (int): Mask size for Fourier filtering.
        min_filter_kernel (int): Kernel size for MinFilter.
        max_filter_kernel (int): Kernel size for MaxFilter.
    """
    image = load_image(image_path)
    normalized_image = z_score_normalisation(image)
    filters = [
        ("GaussianBlur", {"ksize": (gaussian_params[0], gaussian_params[1]), "sigma": gaussian_params[2]}),
        ("MedianBlur", {"ksize": median_ksize}),
        ("BilateralFilter", {"d": bilateral_params[0], "sigma_color": bilateral_params[1], "sigma_space": bilateral_params[2]}),
        ("LaplacianFilter", {"ddepth": laplacian_ddepth}),
        ("SobelFilter", {"dx": sobel_params[0], "dy": sobel_params[1], "ksize": sobel_params[2]}),
        ("CannyEdge", {"threshold1": canny_params[0], "threshold2": canny_params[1]}),
        ("AdaptiveThreshold", {"max_value": adaptive_threshold_params[0], "block_size": adaptive_threshold_params[1], "c": adaptive_threshold_params[2]}),
        ("TopHat", {"kernel_size": top_hat_kernel}),
        ("BlackHat", {"kernel_size": black_hat_kernel}),
        ("GaborFilter", {"ksize": gabor_params[0], "sigma": gabor_params[1], "theta": gabor_params[2], "lambd": gabor_params[3], "gamma": gabor_params[4]}),
        ("FourierFilter", {"mask_size": fourier_mask_size}),
        ("MinFilter", {"kernel_size": min_filter_kernel}),
        ("MaxFilter", {"kernel_size": max_filter_kernel}),
    ]
    results = []
    for filter_name, params in filters:
        filtered_image = apply_filter(normalized_image, filter_name, **params)
        histograms = {
            "Original": calculate_histogram(image),
            "Normalized": calculate_histogram(normalized_image),
            "Filtered": calculate_histogram(filtered_image),
        }
        results.append((filter_name, image, normalized_image, filtered_image, histograms))
    visualize_results(results)
