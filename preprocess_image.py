import os
import argparse
import cv2
import numpy as np

def color_normalization_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def skin_color_filter_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

def edge_enhancement_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def denoise_median(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def histogram_equalization(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def smoothing_gaussian(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


# ------ Additional Functions for Image Processing ----

def contrast_enhancement(image):
    """Apply histogram equalization for contrast enhancement."""
    # Regular histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    equalized = cv2.equalizeHist(gray)
    
    # If original is color, merge equalized luminance with color channels
    if len(image.shape) == 3:
        # Convert to LAB, replace L channel, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = equalized
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
    return equalized

def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    if len(image.shape) == 3:
        # Convert to LAB for better color handling
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

def color_space_transform(image, space='hsv'):
    """Convert image to different color spaces to enhance skin detection."""
    if space.lower() == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif space.lower() == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif space.lower() == 'ycrcb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError("Unsupported color space. Use 'hsv', 'lab', or 'ycrcb'")

def edge_enhancement(image):
    """Enhance edges in the image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Add edge map to original image for enhancement
    if len(image.shape) == 3:
        # Create 3-channel edge map
        edge_map = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
        # Blend with original with weight
        enhanced = cv2.addWeighted(image, 0.7, edge_map, 0.3, 0)
        return enhanced
    else:
        # For grayscale, just blend directly
        enhanced = cv2.addWeighted(gray, 0.7, magnitude, 0.3, 0)
        return enhanced

def unsharp_masking(image, sigma=1.0, strength=1.5):
    """Apply unsharp masking to enhance edges."""
    if len(image.shape) == 3:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    else:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened

def noise_reduction(image, method='gaussian', kernel_size=5):
    """Apply noise reduction."""
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        # Bilateral filter preserves edges better
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError("Unsupported method. Use 'gaussian', 'median', or 'bilateral'")

def morphological_operations(image, operation='opening', kernel_size=5):
    """Apply morphological operations."""
    if len(image.shape) == 3:
        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'opening':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilation':
        result = cv2.dilate(gray, kernel, iterations=1)
    elif operation == 'erosion':
        result = cv2.erode(gray, kernel, iterations=1)
    else:
        raise ValueError("Unsupported operation. Use 'opening', 'closing', 'dilation', or 'erosion'")
    
    # If original is color, convert result back to color
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def image_normalization(image, method='minmax'):
    """Normalize image pixel values."""
    if method == 'minmax':
        # Min-max scaling to [0, 1]
        if len(image.shape) == 3:
            normalized = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                channel = image[:,:,i].astype(np.float32)
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val > min_val:
                    normalized[:,:,i] = (channel - min_val) / (max_val - min_val)
            return (normalized * 255).astype(np.uint8)
        else:
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized = (image.astype(np.float32) - min_val) / (max_val - min_val)
                return (normalized * 255).astype(np.uint8)
            return image
    elif method == 'standardize':
        # Standardization (zero mean, unit variance)
        if len(image.shape) == 3:
            normalized = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                channel = image[:,:,i].astype(np.float32)
                mean = np.mean(channel)
                std = np.std(channel)
                if std > 0:
                    normalized[:,:,i] = (channel - mean) / std
            # Rescale to 0-255 range for visualization
            normalized = normalized - np.min(normalized)
            normalized = normalized / np.max(normalized) * 255
            return normalized.astype(np.uint8)
        else:
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                normalized = (image.astype(np.float32) - mean) / std
                # Rescale to 0-255 range for visualization
                normalized = normalized - np.min(normalized)
                normalized = normalized / np.max(normalized) * 255
                return normalized.astype(np.uint8)
            return image
    else:
        raise ValueError("Unsupported method. Use 'minmax' or 'standardize'")

def skin_detection_roi(image):
    """Detect skin regions as a potential ROI for hands."""
    # Convert to YCrCb color space which is good for skin detection
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color bounds in YCrCb space
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create binary mask of skin pixels
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to original image
    skin_only = cv2.bitwise_and(image, image, mask=skin_mask)
    
    return skin_only, skin_mask

def illumination_correction(image, gamma=1.0):
    """Apply gamma correction for illumination correction."""
    # Gamma correction
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    
    return gamma_corrected.astype(np.uint8)

def retinex_filtering(image, sigma_list=[15, 80, 250]):
    """Apply Multi-Scale Retinex (MSR) for better visibility in shadows/bright areas."""
    if len(image.shape) == 3:
        # Process each channel
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            channel = image[:,:,i].astype(np.float32)
            retinex = np.zeros_like(channel)
            
            # Multi-scale Retinex
            for sigma in sigma_list:
                # Gaussian filter
                blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
                # Add small value to avoid log(0)
                blurred = np.where(blurred < 1.0, 1.0, blurred)
                # Single-scale Retinex
                retinex += np.log10(channel + 1.0) - np.log10(blurred)
                
            # Normalize to 0-255 range
            retinex = retinex / len(sigma_list)
            retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255.0
            result[:,:,i] = retinex
            
        return result.astype(np.uint8)
    else:
        image = image.astype(np.float32)
        retinex = np.zeros_like(image)
        
        # Multi-scale Retinex
        for sigma in sigma_list:
            # Gaussian filter
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            # Add small value to avoid log(0)
            blurred = np.where(blurred < 1.0, 1.0, blurred)
            # Single-scale Retinex
            retinex += np.log10(image + 1.0) - np.log10(blurred)
            
        # Normalize to 0-255 range
        retinex = retinex / len(sigma_list)
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255.0
        
        return retinex.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Process images for human pose estimation')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output_dir', default='output', help='Directory to save results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all images in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, filename)
            print(f"Processing {filename}...")
            
            # Read the image
            image = cv2.imread(image_path)

            # Apply each correction function and save the result
            # output_clahe = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_clahe.jpg")
            # cv2.imwrite(output_clahe, color_normalization_clahe(image))

            # output_skin_filter = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_skin_filter.jpg")
            # cv2.imwrite(output_skin_filter, skin_color_filter_hsv(image))

            # output_edge_enhanced = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_edge_enhanced.jpg")
            # cv2.imwrite(output_edge_enhanced, edge_enhancement_laplacian(image))

            # output_denoised = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_denoised.jpg")
            # cv2.imwrite(output_denoised, denoise_median(image))

            # output_hist_eq = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_hist_eq.jpg")
            # cv2.imwrite(output_hist_eq, histogram_equalization(image))

            # output_smoothed = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_smoothed.jpg")
            # cv2.imwrite(output_smoothed, smoothing_gaussian(image))
    
            # output_contrast = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_contrast.jpg")
            # cv2.imwrite(output_contrast, contrast_enhancement(image))

            # output_clahe_applied = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_clahe_applied.jpg")
            # cv2.imwrite(output_clahe_applied, apply_clahe(image))

            # output_color_space = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_color_space.jpg")
            # cv2.imwrite(output_color_space, color_space_transform(image, space='hsv'))

            # output_edge_enhanced = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_edge_enhanced.jpg")
            # cv2.imwrite(output_edge_enhanced, edge_enhancement(image))

            # output_unsharp = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_unsharp.jpg")
            # cv2.imwrite(output_unsharp, unsharp_masking(image))

            # output_noise_reduced = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_noise_reduced.jpg")
            # cv2.imwrite(output_noise_reduced, noise_reduction(image, method='bilateral'))

            # output_morph = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_morph.jpg")
            # cv2.imwrite(output_morph, morphological_operations(image, operation='opening'))

            # output_normalized = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_normalized.jpg")
            # cv2.imwrite(output_normalized, image_normalization(image, method='minmax'))

            # output_skin_detection = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_skin_detection.jpg")
            # skin_only, _ = skin_detection_roi(image)
            # cv2.imwrite(output_skin_detection, skin_only)

            # output_illumination = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_illumination.jpg")
            # cv2.imwrite(output_illumination, illumination_correction(image, gamma=1.2))

            # output_retinex = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_retinex.jpg")
            # cv2.imwrite(output_retinex, retinex_filtering(image))
            
            for operation in ['opening', 'closing', 'dilation', 'erosion']:
                output_morph = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_morph_{operation}.jpg")
                cv2.imwrite(output_morph, morphological_operations(image, operation=operation))
    
if __name__ == "__main__":
    main()