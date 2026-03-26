"""
Image Clustering Pipeline - Feature Extraction and Clustering

This module implements a comprehensive clustering pipeline for image analysis that:
1. Loads images from a specified directory
2. Computes various feature descriptors (ResNet50, DINO-v2, GLCM, SIFT, HOG, Gray Histogram)
3. Applies clustering algorithms (K-Means, Spectral, GMM, Agglomerative)
4. Computes clustering metrics and silhouette scores
5. Exports results in Excel and CSV formats for dashboard visualization

Usage:
    python pipeline.py --path_data /path/to/data --path_output /path/to/output
    
    Arguments:
        --path_data: Path to directory containing images (required)
        --path_output: Path to output directory for results (required)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import threading
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture   
from sklearn.decomposition import PCA

# Import from modular structure
from features import (
    compute_dinov2_descriptors,
    compute_resnet50_descriptors,
    compute_gray_histograms_base_images,
    compute_hog_descriptors_base_images,
    compute_sift_descriptors,
    compute_glcm_descriptors_base_images,
)
from clustering import KMeans, show_metric, compute_silhouette_sweep, save_sweep_results
from utils import image_loader, conversion_3d, create_df_to_export
from config import PATH_OUTPUT


class Spinner:
    """
    Terminal spinner with message support for progress indication.
    
    Displays an animated spinner while allowing status messages to be
    printed without disrupting the animation. Thread-safe for use with
    long-running operations.
    
    Attributes:
        busy (bool): Whether spinner is currently running
        chars (list): Characters used for animation frames
        index (int): Current animation frame index
        paused (bool): Whether spinner is temporarily paused for output
    
    Example:
        >>> spinner = Spinner()
        >>> spinner.start()
        >>> spinner.message("Processing...")
        >>> spinner.stop()
    """
    def __init__(self):
        self.busy = False
        self.chars = ['|', '/', '-', '\\']
        self.index = 0
        self.paused = False
        
    def start(self):
        """Start the spinner animation in a background thread."""
        self.busy = True
        self.paused = False
        threading.Thread(target=self._spin, daemon=True).start()
        
    def _spin(self):
        """Internal method: animate the spinner."""
        while self.busy:
            if not self.paused:
                sys.stdout.write(self.chars[self.index % 4])
                sys.stdout.flush()
                sys.stdout.write('\b')
                time.sleep(0.1)
                self.index += 1
            else:
                time.sleep(0.05)
            
    def pause(self):
        """Pause the spinner for printing messages."""
        self.paused = True
        time.sleep(0.15)
        sys.stdout.write(' \b')
        sys.stdout.flush()
        
    def resume(self):
        """Resume the spinner after printing."""
        self.paused = False
        
    def message(self, msg: str):
        """
        Print a message while pausing the spinner.
        
        Args:
            msg: Message to print
        """
        self.pause()
        print(msg)
        self.resume()
            
    def stop(self):
        """Stop the spinner animation."""
        self.busy = False
        time.sleep(0.15)
        sys.stdout.write(' ')
        sys.stdout.flush()
        sys.stdout.write('\b')


FEATURES = [
    "resnet50",
    "dinov2", 
    "gray_histogram", 
    "hog", 
    "sift",
    "glcm"
    ]  # add new features here
MODELS = [
    "kmeans", 
    "spectral", 
    "gmm_diag",
    "agglomerative"
    ]  # add new clustering models here


def _make_output_filenames(feature: str, model: str) -> tuple[str, str, str, str]:
    """
    Generate output filenames for clustering results and metrics.
    
    Args:
        feature: Feature descriptor name (e.g., 'resnet50', 'dinov2')
        model: Clustering model name (e.g., 'kmeans', 'spectral')
        
    Returns:
        Tuple of (clustering_xlsx, clustering_csv, metric_xlsx, metric_csv)
    """
    feature_key = feature.lower()
    model_key = model.lower()
    clustering_xlsx = f"save_clustering__{feature_key}__{model_key}.xlsx"
    clustering_csv = f"save_clustering__{feature_key}__{model_key}.csv"
    metric_xlsx = f"save_metric__{feature_key}__{model_key}.xlsx"
    metric_csv = f"save_metric__{feature_key}__{model_key}.csv"
    return clustering_xlsx, clustering_csv, metric_xlsx, metric_csv


def _safe_pca_transform(X, n_components: int, *, random_state: int = 42):
    """
    Apply PCA only when valid. Returns (X_out, pca_or_None).
    PCA constraint: n_components <= min(n_samples - 1, n_features).
    """
    if n_components is None:
        return X, None

    n_samples, n_features = X.shape
    max_components = min(n_features, max(n_samples - 1, 0))
    if max_components < 2:
        # too small to PCA
        return X, None

    n_comp = min(int(n_components), int(max_components))
    if n_comp < 2:
        return X, None

    pca = PCA(n_components=n_comp, whiten=False, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def _compute_feature_descriptors(feature: str, base_images: list[dict]) -> np.ndarray:
    """
    Compute descriptors for a single feature.
    
    Args:
        feature: Feature descriptor name
        base_images: List of image dictionaries
        
    Returns:
        Descriptor array as float32 numpy array
    """
    if feature == "resnet50":
        descriptors = compute_resnet50_descriptors(base_images)
    elif feature == "dinov2":
        descriptors = compute_dinov2_descriptors(base_images)
    elif feature == "gray_histogram":
        descriptors = compute_gray_histograms_base_images(base_images, n_bins=16)
    elif feature == "hog":
        descriptors = compute_hog_descriptors_base_images(base_images)
    elif feature == "sift":
        descriptors = compute_sift_descriptors(base_images)
    elif feature == "glcm":
        descriptors = compute_glcm_descriptors_base_images(base_images)
    else:
        raise ValueError(f"Unknown feature: {feature}")
    
    return np.asarray(descriptors, dtype=np.float32)


def _preprocess_descriptors(feature: str, descriptors, pca_components: int):
    """
    Apply feature-specific preprocessing to descriptors.
    
    Preprocessing pipeline:
    1. StandardScaler: Applied to features with heterogeneous scales
       (HOG, SIFT, GLCM) to normalize feature magnitudes
    2. PCA: Dimensionality reduction for certain feature types
       - Skipped for gray_histogram (already low-dimensional)
       - Skipped for GLCM (texture features preserve structure better without PCA)
       - Applied to deep learning features (ResNet50, DINO-v2) for efficiency
    3. L2 normalization: Final step to normalize feature vectors
    
    Args:
        feature: Feature descriptor name
        descriptors: Input descriptor array
        pca_components: Target number of PCA components
        
    Returns:
        Tuple of (normalized_descriptors, pca_model_or_None)
        
    Note:
        - gray_histogram: Low-dimensional, uses only L2 normalization
        - hog/sift/glcm: Applied StandardScaler + optional PCA + L2
        - dinov2/resnet50: Applied PCA + L2 for efficiency
    """
    X = descriptors

    # StandardScaler for features with heterogeneous scales
    if feature in ("hog", "sift", "glcm"):
        X = StandardScaler().fit_transform(X)

    # Skip PCA for low-dimensional descriptors
    if feature in ("gray_histogram", "glcm"):
        X_pca = X
        pca = None
    else:
        X_pca, pca = _safe_pca_transform(X, pca_components)

    X_norm = normalize(X_pca, norm="l2")
    return X_norm, pca


def _run_clustering(
    *,
    feature: str,
    model: str,
    descriptors_norm: np.ndarray,
    pca,
    base_images: list[dict],
    labels_true: list[str],
    labels_true_encoded,
    path_output: str,
):
    """
    Run clustering on preprocessed descriptors (assumes descriptors already computed).
    
    Performs: clustering → metrics computation → export to Excel and CSV.
    
    Args:
        feature: Feature descriptor name
        model: Clustering model name
        descriptors_norm: Preprocessed normalized descriptors
        pca: PCA model or None
        base_images: List of image dictionaries
        labels_true: True labels for images
        labels_true_encoded: Encoded integer labels
        path_output: Output directory path
    """
    spinner = Spinner()
    spinner.start()
    spinner.message(f"◆ Clustering [{feature}/{model}]...")

    # -------- PCA info --------
    if pca is not None:
        explained = float(pca.explained_variance_ratio_.sum())
        used_components = int(pca.n_components_)
        spinner.message(f"▶ [{feature}/{model}] PCA components used: {used_components}")
        spinner.message(f"▶ [{feature}/{model}] explained variance sum: {explained:.4f}")
    else:
        explained = None
        used_components = None
        spinner.message(f"▶ [{feature}/{model}] PCA skipped")

    # -------- Clustering --------
    number_cluster = len(set(labels_true_encoded))

    if model == "kmeans":
        clusterer = KMeans(
            n_clusters=number_cluster,
            max_iter=300,
            n_init=20,
            random_state=42,
            init="k-means++",
        )
        clusterer.fit(descriptors_norm)
        labels_pred = clusterer.labels_

    elif model == "spectral":
        n_neighbors = min(20, descriptors_norm.shape[0] - 1)

        clusterer = SpectralClustering(
            n_clusters=number_cluster,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=42,
        )
        labels_pred = clusterer.fit_predict(descriptors_norm)

    elif model == "gmm_diag":
        clusterer = GaussianMixture(
            n_components=number_cluster,
            covariance_type="diag",
            n_init=5,
            max_iter=300,
            random_state=42,
        )
        clusterer.fit(descriptors_norm)
        labels_pred = clusterer.predict(descriptors_norm)

    elif model == "agglomerative":
        clusterer = AgglomerativeClustering(
            n_clusters=number_cluster,
            linkage="ward",
        )
        labels_pred = clusterer.fit_predict(descriptors_norm)

    else:
        raise ValueError(f"Unknown model: {model}")

    spinner.message(f"▶ [{feature}/{model}] Clustering complete, computing metrics...")

    # -------- Silhouette sweep (for dashboard graph) --------
    spinner.message(f"▶ [{feature}/{model}] Computing silhouette sweep...")
    sweep_results = compute_silhouette_sweep(feature, model, descriptors_norm, path_output)
    save_sweep_results(feature, model, sweep_results, path_output)
    spinner.message(f"▶ [{feature}/{model}] Silhouette sweep saved")

    # -------- Metrics --------
    metric = show_metric(
        labels_true_encoded,
        labels_pred,
        descriptors_norm,
        bool_show=True,
        name_descriptor=feature.upper(),
        name_model=model,
        bool_return=True,
    )

    # metadata for dashboard
    metric["feature"] = feature
    metric["model"] = model
    metric["explained_variance_sum"] = explained

    # -------- Export for dashboard --------
    x_3d = conversion_3d(descriptors_norm)

    df_cluster = create_df_to_export(
        x_3d,
        labels_true,
        labels_pred,
        base_images,
    )

    clustering_xlsx, clustering_csv, metric_xlsx, metric_csv = _make_output_filenames(feature, model)

    os.makedirs(path_output, exist_ok=True)
    
    # Export clustering results
    df_cluster.to_excel(os.path.join(path_output, clustering_xlsx), index=False)
    df_cluster.to_csv(os.path.join(path_output, clustering_csv), index=False)
    
    # Export metrics
    pd.DataFrame([metric]).to_excel(os.path.join(path_output, metric_xlsx), index=False)
    pd.DataFrame([metric]).to_csv(os.path.join(path_output, metric_csv), index=False)

    spinner.stop()
    print(f"✓ [{feature}/{model}] wrote: {clustering_xlsx}, {clustering_csv}")
    print(f"✓ [{feature}/{model}] wrote: {metric_xlsx}, {metric_csv}")


def pipeline(path_data: str, path_output: str, pca_components: int = 32):
    """
    Execute the complete clustering pipeline
    
    Workflow:
    1. Loads images from the specified data directory
    2. Computes descriptors once per feature
    3. Applies all clustering models to cached descriptors
    4. Exports results to Excel and CSV formats
    
    Args:
        path_data: Path to directory containing images organized by class
        path_output: Path to output directory for results
        pca_components: Number of PCA components for preprocessing (default: 32)
    """
    print("=" * 60)
    print("IMAGE CLUSTERING PIPELINE (Optimized - Descriptor Caching)")
    print("=" * 60)
    print(f"Data path: {path_data}")
    print(f"Output path: {path_output}")
    print("=" * 60)
    
    print("\n##########   LOADING IMAGES  ##########")
    base_images, labels_true = image_loader(path_data)
    print(f"✓ Loaded {len(base_images)} images with {len(set(labels_true))} classes")

    # Convert string labels to integers for compatibility with clustering metrics
    label_encoder = LabelEncoder()
    labels_true_encoded = label_encoder.fit_transform(labels_true)

    # Step 1: Compute descriptors once per feature and cache them
    print("\n##########   COMPUTING DESCRIPTORS  ##########")
    descriptors_cache = {}
    pca_cache = {}
    
    for feature in FEATURES:
        spinner = Spinner()
        spinner.start()
        spinner.message(f"▶ Computing {feature}...")
        
        # Compute raw descriptors
        descriptors = _compute_feature_descriptors(feature, base_images)
        spinner.message(f"▶ {feature} shape: {descriptors.shape}")
        
        # Preprocess once
        descriptors_norm, pca = _preprocess_descriptors(feature, descriptors, pca_components)
        
        # Cache for all models
        descriptors_cache[feature] = descriptors_norm
        pca_cache[feature] = pca
        
        if pca is not None:
            explained = float(pca.explained_variance_ratio_.sum())
            spinner.message(f"✓ {feature} preprocessed (variance: {explained:.4f})")
        else:
            spinner.message(f"✓ {feature} preprocessed (PCA skipped)")
        
        spinner.stop()
    
    print(f"✓ All {len(FEATURES)} descriptors computed and cached")

    # Step 2: Run all clustering models with cached descriptors
    print("\n##########   CLUSTERING  ##########")
    total_combinations = len(FEATURES) * len(MODELS)
    current = 0
    
    for feature in FEATURES:
        descriptors_norm = descriptors_cache[feature]
        pca = pca_cache[feature]
        
        for model in MODELS:
            current += 1
            print(f"\n[{current}/{total_combinations}] Clustering {feature} + {model}...")
            _run_clustering(
                feature=feature,
                model=model,
                descriptors_norm=descriptors_norm,
                pca=pca,
                base_images=base_images,
                labels_true=labels_true,
                labels_true_encoded=labels_true_encoded,
                path_output=path_output,
            )

    print("\n" + "=" * 60)
    print("[SUCCESS] Pipeline completed successfully!")
    print(f"Results saved to: {path_output}")
    print("=" * 60)
    print("\nTo visualize results, run:")
    print(f"  python dashboard.py --path_data {path_output}\n")


def main():
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Image Clustering Pipeline - Extract features and perform clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --path_data ./images/data/test --path_output ./output
  python pipeline.py -d /path/to/images -o /path/to/output
        """
    )
    
    parser.add_argument(
        "--path_data", "-d",
        required=True,
        type=str,
        help="Path to directory containing images (organized by class subdirectories)"
    )
    parser.add_argument(
        "--path_output", "-o",
        required=True,
        type=str,
        help="Path to output directory for clustering results (Excel and CSV files)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.path_data):
        print(f"[ERROR] Error: Data path does not exist: {args.path_data}")
        sys.exit(1)
    
    # Convert to absolute paths
    path_data = os.path.abspath(args.path_data)
    path_output = os.path.abspath(args.path_output)
    
    # Run pipeline
    try:
        pipeline(path_data, path_output)
    except Exception as e:
        print(f"\n[ERROR] Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()