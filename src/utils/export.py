"""Data export utilities for clustering results."""

import pandas as pd
from typing import Optional, List


def create_df_to_export(
    data_3d: list,
    l_true_label: List[str],
    l_cluster: List[int],
    base_images: Optional[List[dict]] = None
) -> pd.DataFrame:
    """
    Create DataFrame from clustering results for export to Excel/CSV.
    
    Combines 3D visualization coordinates, true labels, cluster assignments,
    and image paths into a structured DataFrame.
    
    Args:
        data_3d: 3D coordinates from t-SNE or other dimensionality reduction
                array-like of shape (n_samples, 3)
        l_true_label: True class labels (ground truth)
        l_cluster: Predicted cluster assignments
        base_images: Optional list of base image dicts containing 'path' keys
    
    Returns:
        pd.DataFrame with columns: x, y, z (coordinates), label, cluster,
                                  optionally image_path
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    
    # Add image paths if available
    if base_images is not None:
        df['image_path'] = [img.get('path', '') for img in base_images]
    
    return df
