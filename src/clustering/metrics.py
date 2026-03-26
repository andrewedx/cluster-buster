"""Clustering evaluation metrics."""

from __future__ import annotations

from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)
import numpy as np


def show_metric(
    labels_true,
    labels_pred,
    descriptors,
    bool_return: bool = False,
    name_descriptor: str = "",
    name_model: str = "",
    bool_show: bool = True,
) -> dict | None:
    """
    Compute and display clustering evaluation metrics.
    
    Evaluates clustering quality using multiple metrics that measure:
    - How well the clustering matches the ground truth labels
    - Internal cohesion of clusters
    - Separation between clusters
    
    Args:
        labels_true: True labels of the data (ground truth)
        labels_pred: Predicted cluster assignments
        descriptors: Feature descriptors used for clustering (for silhouette score)
        bool_return: If True, return metrics dictionary; if False, only print
        name_descriptor: Name of the feature descriptor (for display)
        name_model: Name of the clustering model (for display)
        bool_show: If True, print metrics to console
        
    Returns:
        Dictionary with metrics if bool_return=True, else None
        
    Metrics Computed:
        - ARI (Adjusted Rand Index): Similarity between true and predicted labels (-1 to 1)
        - AMI (Adjusted Mutual Information): Mutual information normalized for chance
        - Silhouette Score: How similar objects are to their cluster (-1 to 1)
        - Homogeneity: Whether all samples of a class belong to one cluster
        - Completeness: Whether all samples of a class are grouped together
        - V-measure: Harmonic mean of homogeneity and completeness
    """
    # Compute metrics
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    
    # Display results if requested
    if bool_show:
        print(f"########## Metrics for descriptor: {name_descriptor}")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Homogeneity: {homogeneity:.4f}")
        print(f"Completeness: {completeness:.4f}")
        print(f"V-measure: {v_measure:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Adjusted Mutual Information (AMI): {ami:.4f}")
    
    # Return metrics if requested
    if bool_return:
        return {
            "ami": ami,
            "ari": ari,
            "silhouette": silhouette,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "descriptor": name_descriptor,
            "name_model": name_model,
        }
    
    return None
