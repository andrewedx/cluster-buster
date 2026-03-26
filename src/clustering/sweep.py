"""Silhouette score sweep analysis for multiple cluster counts."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from .kmeans import KMeans


CLUSTER_COUNTS = [5, 10, 15, 20, 25]


def get_sweep_filename(feature: str, model: str) -> str:
    """
    Generate filename for sweep results cache.
    
    Args:
        feature: Feature name
        model: Clustering model name
        
    Returns:
        Filename following naming convention: sweep_silhouette__<feature>__<model>.json
    """
    feature_key = feature.lower()
    model_key = model.lower()
    return f"sweep_silhouette__{feature_key}__{model_key}.json"


def compute_silhouette_sweep(
    feature: str,
    model: str,
    descriptors_norm: np.ndarray,
    output_dir: str,
) -> dict:
    """
    Compute silhouette scores for multiple cluster counts.
    
    Tests clustering quality across different cluster numbers to help identify
    the optimal number of clusters. Higher silhouette scores indicate better
    cluster separation and cohesion.
    
    Args:
        feature: Feature descriptor name
        model: Clustering model name
        descriptors_norm: Normalized descriptor array of shape (n_samples, n_features)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with cluster counts as string keys and silhouette scores as float values
    """
    results = {}
    
    for n_clusters in CLUSTER_COUNTS:
        # Ensure n_clusters doesn't exceed number of samples
        if n_clusters > len(descriptors_norm):
            continue
            
        try:
            if model == "kmeans":
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    max_iter=300,
                    n_init=20,
                    random_state=42,
                    init="k-means++",
                )
                clusterer.fit(descriptors_norm)
                labels = clusterer.labels_
                
            elif model == "spectral":
                n_neighbors = min(20, len(descriptors_norm) - 1)
                clusterer = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    assign_labels="kmeans",
                    random_state=42,
                )
                labels = clusterer.fit_predict(descriptors_norm)
                
            elif model == "gmm_diag":
                clusterer = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="diag",
                    n_init=5,
                    max_iter=300,
                    random_state=42,
                )
                clusterer.fit(descriptors_norm)
                labels = clusterer.predict(descriptors_norm)
                
            elif model == "gmm_full":
                clusterer = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    n_init=5,
                    max_iter=300,
                    random_state=42,
                )
                clusterer.fit(descriptors_norm)
                labels = clusterer.predict(descriptors_norm)
                
            elif model == "agglomerative":
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="ward",
                )
                labels = clusterer.fit_predict(descriptors_norm)
                
            else:
                raise ValueError(f"Unknown model: {model}")
            
            # Compute silhouette score
            score = float(silhouette_score(descriptors_norm, labels))
            results[str(n_clusters)] = score
            
        except Exception as e:
            print(f"Warning: Failed to compute silhouette for {n_clusters} clusters: {e}")
            continue
    
    return results


def save_sweep_results(
    feature: str,
    model: str,
    sweep_results: dict,
    output_dir: str,
) -> None:
    """
    Save sweep results to JSON file.
    
    Args:
        feature: Feature descriptor name
        model: Clustering model name
        sweep_results: Dictionary of sweep results
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = get_sweep_filename(feature, model)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(sweep_results, f, indent=2)


def load_sweep_results(
    feature: str,
    model: str,
    output_dir: str,
) -> dict | None:
    """
    Load sweep results from JSON file.
    
    Args:
        feature: Feature descriptor name
        model: Clustering model name
        output_dir: Output directory path
        
    Returns:
        Dictionary of sweep results or None if file not found
    """
    filename = get_sweep_filename(feature, model)
    filepath = os.path.join(output_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load sweep results: {e}")
        return None
