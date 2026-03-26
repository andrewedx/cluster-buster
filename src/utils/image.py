"""Image loading and preprocessing utilities."""

import os
import numpy as np
from PIL import Image, ImageOps


def image_loader(datasetPath):
    """
    Load images from a directory structure organized by class subdirectories.
    
    Applies global preprocessing: EXIF rotation correction, RGB conversion,
    and normalization to float32 [0,1] range.
    
    Args:
        datasetPath: Path to directory containing class subdirectories
                    Expected structure: datasetPath/class1/img1.jpg, class2/img2.jpg, etc.
    
    Returns:
        Tuple of (base_images, labels) where:
        - base_images: List of dicts with keys:
            - 'data': H x W x 3 float32 RGB in [0,1]
            - 'width': Image width
            - 'height': Image height
            - 'path': Full path to image file
            - 'label_name': Class name
        - labels: List of class names corresponding to each image
    """
    base_images = []
    labels = []
    category_counts = {}

    for label in os.listdir(datasetPath):
        label_path = os.path.join(datasetPath, label)

        if not os.path.isdir(label_path):
            continue

        category_counts[label] = 0

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            try:
                # Load image
                pil_img = Image.open(img_path)
                # Correct EXIF rotation
                pil_img = ImageOps.exif_transpose(pil_img)
                # Convert to RGB
                pil_img = pil_img.convert("RGB")
            except Exception as e:
                print(f"Error loading image: {img_path} - {e}")
                continue

            # Convert to float32 RGB in [0,1]
            img_np = np.asarray(pil_img).astype("float32") / 255.0

            # Create base image dictionary
            base_img = {
                "data": img_np,                # H x W x 3 float32
                "width": img_np.shape[1],
                "height": img_np.shape[0],
                "path": img_path,
                "label_name": label
            }

            base_images.append(base_img)
            labels.append(label)
            category_counts[label] += 1

    # Print image counts per category
    print("\n--- Image counts per category ---")
    for category in sorted(category_counts.keys()):
        print(f"{category}: {category_counts[category]} images")
    print("-" * 32 + "\n")

    return base_images, labels
