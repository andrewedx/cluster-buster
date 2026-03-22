import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


def _to_gray_uint8_from_base_image(item: dict) -> np.ndarray:
    """
    base_images[i]["data"] is float32 RGB in [0,1], shape (H,W,3).
    Convert to grayscale uint8 (H,W) in [0,255] for GLCM.
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


def compute_glcm_descriptors_base_images(
    base_images: list[dict],
    *,
    distances: list[int] = None,
    angles: list[float] = None,
    levels: int = 256,
) -> np.ndarray:
    """
    Compute GLCM (Gray Level Co-occurrence Matrix) descriptors for base_images (pipeline-compatible).

    Expects base_images[i]["data"] as float32 RGB in [0,1], shape (H, W, 3).

    Args:
        base_images: List of dictionaries with 'data' key containing float32 RGB images
        distances: List of distances for GLCM computation (default: [1])
        angles: List of angles (in radians) for GLCM computation (default: [0])
        levels: Number of gray levels to quantize to (default: 256)

    Returns:
        np.ndarray of shape (n_images, n_features), float32.
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0]

    if base_images is None or len(base_images) == 0:
        return np.empty((0, len(distances) * len(angles) * 6), dtype=np.float32)

    descriptors = []

    for item in base_images:
        try:
            # Convert to grayscale uint8
            gray = _to_gray_uint8_from_base_image(item)

            # Quantize image to specified number of levels
            gray_quantized = ((gray / 255.0) * (levels - 1)).astype(np.uint8)

            # Compute GLCM
            glcm = graycomatrix(
                gray_quantized,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True,
            )

            # Extract GLCM properties
            features = []
            for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
                prop_values = graycoprops(glcm, prop)
                features.extend(prop_values.flatten())

            descriptors.append(np.array(features, dtype=np.float32))

        except Exception as e:
            print(f"Error processing image: {e}")
            # Add zero vector if processing fails
            descriptors.append(np.zeros(len(distances) * len(angles) * 6, dtype=np.float32))

    return np.vstack(descriptors).astype(np.float32)