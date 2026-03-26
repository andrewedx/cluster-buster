"""Visualization utilities for dimensionality reduction and 3D projection."""

import numpy as np
from sklearn.manifold import TSNE


def conversion_3d(
    X,
    n_components: int = 3,
    perplexity: float = 50,
    random_state: int = 42,
    early_exaggeration: float = 10,
    n_iter: int = 3000
) -> np.ndarray:
    """
    Reduce high-dimensional feature vectors to 3D using t-SNE.
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding) preserves local structure
    and creates visually interpretable 2D/3D projections of high-dimensional data.
    
    Args:
        X: Input data array of shape (n_samples, n_features)
        n_components: Number of output dimensions (default: 3 for 3D visualization)
        perplexity: Balances local and global structure - higher values emphasize global structure (default: 50)
        random_state: Random seed for reproducibility (default: 42)
        early_exaggeration: Scaling factor for early learning - makes clusters more separated (default: 10)
        n_iter: Number of iterations for optimization (default: 3000)
    
    Returns:
        np.ndarray of shape (n_samples, n_components) with projected coordinates
    """
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        max_iter=n_iter
    )
    X = np.array(X)
    X_3d = tsne.fit_transform(X)
    return X_3d
