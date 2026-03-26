"""Feature extraction module - Image feature descriptors.

Provides various feature extraction methods for image clustering:
- ResNet50: ImageNet pre-trained CNN features
- DINO-v2: Vision transformer features
- SIFT: Scale-invariant feature transform
- GLCM: Gray-level co-occurrence matrix (texture)
- Histogram: Gray-level histogram
- HOG: Histogram of oriented gradients
"""

from .neural import compute_resnet50_descriptors, compute_dinov2_descriptors
from .sift import compute_sift_descriptors
from .glcm import compute_glcm_descriptors_base_images
from .histogram import compute_gray_histograms_base_images, compute_hog_descriptors_base_images

__all__ = [
    "compute_resnet50_descriptors",
    "compute_dinov2_descriptors",
    "compute_sift_descriptors",
    "compute_glcm_descriptors_base_images",
    "compute_gray_histograms_base_images",
    "compute_hog_descriptors_base_images",
]
