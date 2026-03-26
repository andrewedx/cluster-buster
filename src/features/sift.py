"""SIFT descriptor extraction with color features."""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def compute_sift_descriptors(base_images, vocab_size=200, img_size=(128, 128)):
    """
    Compute SIFT descriptors + RGB color histogram for image features.
    
    Color features help distinguish food items by their natural colors:
    - Red apples, orange carrots, yellow bananas, etc.

    Args:
        base_images: List of dicts with 'data' key (H x W x 3 float32 in [0,1])
        vocab_size: Number of visual words for Bag-of-Words (default: 200)
        img_size: Target image size for processing (default: 128x128)

    Returns:
        np.ndarray of shape (n_samples, vocab_size + color_bins)
        Combines SIFT Bag-of-Words with color histogram features
    """
    sift = cv2.SIFT_create()
    all_raw_descriptors = []
    per_image_descriptors = []
    color_features_list = []

    print(f"  [SIFT] Extracting from {len(base_images)} images...")
    print(f"  [SIFT] Resize → {img_size} | vocab_size → {vocab_size}")

    # Extract SIFT and color features for each image
    for base_img in base_images:
        img_np = base_img["data"]  # float32 [0,1], H x W x 3 RGB

        # Convert to uint8
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Resize to fixed size
        img_resized = cv2.resize(img_uint8, img_size)

        # SIFT on grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        _, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            per_image_descriptors.append(descriptors)
            all_raw_descriptors.append(descriptors)
        else:
            per_image_descriptors.append(None)

        # RGB color histogram (32 bins per channel)
        color_hist = []
        for canal in range(3):  # R, G, B
            hist, _ = np.histogram(
                img_resized[:, :, canal],
                bins=32,
                range=(0, 256)
            )
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # normalize
            color_hist.extend(hist)

        color_features_list.append(np.array(color_hist))

    # Build visual vocabulary using K-means
    print(f"  [SIFT] Building visual vocabulary ({vocab_size} words)...")
    stacked = np.vstack(all_raw_descriptors)
    print(f"  [SIFT] Total raw descriptors: {stacked.shape}")

    kmeans_vocab = MiniBatchKMeans(
        n_clusters=vocab_size,
        random_state=42,
        batch_size=2048,
        n_init=5
    )
    kmeans_vocab.fit(stacked)

    # Encode each image as Bag-of-Words histogram
    bow_features = []
    for descriptors in per_image_descriptors:
        if descriptors is not None:
            word_ids = kmeans_vocab.predict(descriptors)
            hist, _ = np.histogram(word_ids, bins=np.arange(vocab_size + 1))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
        else:
            hist = np.zeros(vocab_size)
        bow_features.append(hist)

    bow_features = np.vstack(bow_features)
    color_features = np.vstack(color_features_list)

    # Combine SIFT + color (color gets higher weight as it's discriminative for food)
    color_weight = 2.0
    final_features = np.hstack([
        bow_features,
        color_features * color_weight
    ])

    print(f"  [SIFT] SIFT BoW shape: {bow_features.shape}")
    print(f"  [SIFT] Color shape: {color_features.shape}")
    print(f"  [SIFT] Final features: {final_features.shape}")

    return final_features
