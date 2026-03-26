"""Gray-level histogram and HOG (Histogram of Oriented Gradients) feature extraction."""

import numpy as np
import cv2
from skimage.feature import hog


def _to_gray_uint8_from_base_image(item: dict) -> np.ndarray:
    """
    Convert base image to grayscale uint8.
    
    Args:
        item: Dictionary with 'data' key containing float32 RGB in [0,1], shape (H,W,3)
        
    Returns:
        Grayscale uint8 image, shape (H,W) in [0,255]
    """
    x = np.asarray(item["data"])
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {x.shape}")

    # float [0,1] -> uint8 [0,255]
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32, copy=False)
    x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    # RGB -> gray
    gray = cv2.cvtColor(x_u8, cv2.COLOR_RGB2GRAY)
    return gray


def compute_gray_histograms_base_images(base_images: list[dict], n_bins: int = 16) -> np.ndarray:
    """
    Compute gray-level histogram descriptors for images.
    
    A simple and efficient feature that captures the distribution of pixel
    intensities. Useful as a baseline feature for image classification.

    Args:
        base_images: List of dictionaries with 'data' key containing float32 RGB images in [0,1]
        n_bins: Number of histogram bins (default: 16)

    Returns:
        np.ndarray of shape (n_images, n_bins), float32, L1-normalized histograms
    """
    if base_images is None or len(base_images) == 0:
        return np.empty((0, n_bins), dtype=np.float32)

    hist_list = []
    for item in base_images:
        gray = _to_gray_uint8_from_base_image(item)
        hist = cv2.calcHist([gray], [0], None, [n_bins], [0, 256]).astype(np.float32)

        # L1 normalize (sum=1)
        s = float(hist.sum())
        if s > 0:
            hist /= s

        hist_list.append(hist.reshape(-1))

    return np.vstack(hist_list).astype(np.float32)


def compute_hog_descriptors_base_images(
    base_images: list[dict],
    *,
    resize_to: tuple[int, int] = (128, 128),
    orientations: int = 8,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    Compute HOG (Histogram of Oriented Gradients) descriptors for images.
    
    HOG captures edge and shape information through the distribution of gradient
    orientations. Effective for object and shape recognition.
    
    Important: HOG requires consistent image size for all samples, so images
    are resized to `resize_to` dimensions.

    Args:
        base_images: List of dictionaries with 'data' key containing float32 RGB images in [0,1]
        resize_to: Target image size (default: 128x128)
        orientations: Number of gradient orientations (default: 8)
        pixels_per_cell: Size of cells in pixels (default: 8x8)
        cells_per_block: Number of cells per block for normalization (default: 2x2)

    Returns:
        np.ndarray of shape (n_images, descriptor_length), float32
    """
    if base_images is None or len(base_images) == 0:
        return np.empty((0, 0), dtype=np.float32)

    fd_list = []
    for item in base_images:
        gray = _to_gray_uint8_from_base_image(item)

        # Ensure fixed size for consistent descriptor length
        if resize_to is not None:
            gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)

        fd = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True,
        )
        fd_list.append(fd.astype(np.float32))

    return np.vstack(fd_list).astype(np.float32)
