import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filters_to_image(
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
    def z_score_normalisation(image):
        mean, std = np.mean(image), np.std(image)
        normalized = (image - mean) / std
        normalized = ((normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))) * 255
        return normalized.astype(np.uint8)

    def apply_gaussian(image):
        return cv2.GaussianBlur(image, (gaussian_params[0], gaussian_params[1]), gaussian_params[2])

    def apply_median(image):
        return cv2.medianBlur(image, median_ksize)

    def apply_bilateral(image):
        return cv2.bilateralFilter(image, bilateral_params[0], bilateral_params[1], bilateral_params[2])

    def apply_laplacian(image):
        laplacian = cv2.Laplacian(image, laplacian_ddepth)
        return cv2.convertScaleAbs(laplacian)

    def apply_sobel(image):
        return cv2.Sobel(image, cv2.CV_64F, sobel_params[0], sobel_params[1], sobel_params[2])

    def apply_canny(image):
        return cv2.Canny(image, canny_params[0], canny_params[1])

    def apply_adaptive_threshold(image):
        return cv2.adaptiveThreshold(
            image, adaptive_threshold_params[0], cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            adaptive_threshold_params[1], adaptive_threshold_params[2]
        )

    def apply_top_hat(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (top_hat_kernel, top_hat_kernel))
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    def apply_black_hat(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (black_hat_kernel, black_hat_kernel))
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    def apply_gabor(image):
        gabor_kernel = cv2.getGaborKernel(
            (gabor_params[0], gabor_params[0]), gabor_params[1], gabor_params[2], gabor_params[3], gabor_params[4], 0, ktype=cv2.CV_32F
        )
        return cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

    def apply_fourier(image):
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - fourier_mask_size : crow + fourier_mask_size, ccol - fourier_mask_size : ccol + fourier_mask_size] = 1
        filtered_dft = dft_shift * mask
        dft_ishift = np.fft.ifftshift(filtered_dft)
        img_back = cv2.idft(dft_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    def apply_min_filter(image):
        kernel = np.ones((min_filter_kernel, min_filter_kernel), np.uint8)
        return cv2.erode(image, kernel)

    def apply_max_filter(image):
        kernel = np.ones((max_filter_kernel, max_filter_kernel), np.uint8)
        return cv2.dilate(image, kernel)

    filter_functions = {
        "GaussianBlur": apply_gaussian,
        "MedianBlur": apply_median,
        "BilateralFilter": apply_bilateral,
        "LaplacianFilter": apply_laplacian,
        "SobelFilter": apply_sobel,
        "CannyEdge": apply_canny,
        "AdaptiveThreshold": apply_adaptive_threshold,
        "TopHat": apply_top_hat,
        "BlackHat": apply_black_hat,
        "GaborFilter": apply_gabor,
        "FourierFilter": apply_fourier,
        "MinFilter": apply_min_filter,
        "MaxFilter": apply_max_filter,
    }

    def calculate_histogram(image):
        histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
        return histogram

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("HATA: Görsel bulunamadı! Dosya yolunu kontrol edin.")
        return

    normalized_image = z_score_normalisation(image)
    results = []
    for filter_name, filter_function in filter_functions.items():
        filtered_image = filter_function(normalized_image)
        histograms = {
            "Original": calculate_histogram(image),
            "Normalized": calculate_histogram(normalized_image),
            "Filtered": calculate_histogram(filtered_image),
        }
        results.append((filter_name, image, normalized_image, filtered_image, histograms))

    fig, axes = plt.subplots(len(results), 4, figsize=(20, len(results) * 4))
    for idx, (filter_name, original, normalized, filtered, histograms) in enumerate(results):
        axes[idx, 0].imshow(original, cmap="gray")
        axes[idx, 0].set_title(f"Orijinal Görsel\n{filter_name}")
        axes[idx, 1].imshow(normalized, cmap="gray")
        axes[idx, 1].set_title("Z-Score Normalised")
        axes[idx, 2].imshow(filtered, cmap="gray")
        axes[idx, 2].set_title("Filtrelenmiş Görsel")
        axes[idx, 3].plot(histograms["Original"], label="Original", color="blue")
        axes[idx, 3].plot(histograms["Normalized"], label="Normalized", color="green")
        axes[idx, 3].plot(histograms["Filtered"], label="Filtered", color="red")
        axes[idx, 3].legend()
        axes[idx, 3].set_title("Histogram Karşılaştırması")
    plt.tight_layout()
    plt.show()

apply_filters_to_image(example_images/1.jpeg)
