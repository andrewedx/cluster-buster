"""Custom K-Means clustering implementation."""

from __future__ import annotations

import numpy as np


class KMeans:
    """
    Custom K-Means clustering algorithm with K-Means++ initialization.
    
    Implements the full K-Means algorithm with options for initialization
    strategies and multiple runs for robustness.
    
    Attributes:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        n_init: Number of times to run with different initializations
        init: Initialization method ('random' or 'k-means++')
        cluster_centers_: Learned cluster centers
        labels_: Cluster assignments for training data
        inertia_: Sum of squared distances to nearest cluster
    """
    
    def __init__(self, n_clusters=8, max_iter=300, random_state=None, n_init=10, init="k-means++"):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations per run
            random_state: Random seed for reproducibility
            n_init: Number of initialization attempts
            init: Initialization method ('random' or 'k-means++')
        """
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.init = init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _rng(self):
        """Get random number generator."""
        return np.random.default_rng(self.random_state)

    def _init_centers_random(self, X, rng):
        """Random initialization: select random data points as cluster centers."""
        n = X.shape[0]
        replace = n < self.n_clusters
        idx = rng.choice(n, size=self.n_clusters, replace=replace)
        return X[idx].copy()

    def _init_centers_kmeanspp(self, X, rng):
        """
        K-Means++ initialization: select centers with probability proportional to distance squared.
        
        This reduces the number of iterations needed for convergence.
        """
        n, d = X.shape
        centers = np.empty((self.n_clusters, d), dtype=X.dtype)

        # Pick first center uniformly at random
        idx0 = rng.integers(0, n)
        centers[0] = X[idx0]

        # Track squared distance to closest center for each point
        closest_dist_sq = ((X - centers[0]) ** 2).sum(axis=1)

        for k in range(1, self.n_clusters):
            # Select next center with probability proportional to distance^2
            probs = closest_dist_sq / np.sum(closest_dist_sq)
            idx = rng.choice(n, p=probs)
            centers[k] = X[idx]

            # Update distances to closest center
            dist_sq_new_center = ((X - centers[k]) ** 2).sum(axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new_center)

        return centers

    def _init_centers(self, X, rng):
        """Initialize cluster centers using specified method."""
        if self.init == "random":
            return self._init_centers_random(X, rng)
        if self.init == "k-means++":
            return self._init_centers_kmeanspp(X, rng)
        raise ValueError("init must be 'random' or 'k-means++'")

    @staticmethod
    def _assign_labels(X, centers):
        """Assign each point to the nearest cluster center."""
        # Compute squared euclidean distances
        dist_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return dist_sq.argmin(axis=1)

    @staticmethod
    def _compute_centers(X, labels, n_clusters):
        """Compute new cluster centers as mean of assigned points."""
        d = X.shape[1]
        centers = np.empty((n_clusters, d), dtype=X.dtype)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = np.nan  # Will be handled by re-seeding
        return centers

    @staticmethod
    def _compute_inertia(X, centers, labels):
        """Compute inertia: sum of squared distances to nearest center."""
        diff = X - centers[labels]
        return float(np.sum(diff * diff))

    def fit(self, X):
        """
        Fit K-Means clustering to data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        rng_master = self._rng()

        best_inertia = np.inf
        best_centers = None
        best_labels = None

        # Run K-Means multiple times with different initializations
        for run in range(self.n_init):
            # Different seed per run, reproducible if random_state provided
            seed = None if self.random_state is None else int(rng_master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(seed)

            centers = self._init_centers(X, rng)

            # Iterative refinement
            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centers)
                new_centers = self._compute_centers(X, labels, self.n_clusters)

                # Handle empty clusters by re-seeding them to random points
                if np.isnan(new_centers).any():
                    empty = np.isnan(new_centers).any(axis=1)
                    idx = rng.choice(X.shape[0], size=int(empty.sum()), replace=False)
                    new_centers[empty] = X[idx]

                # Check for convergence
                if np.allclose(new_centers, centers):
                    break
                centers = new_centers

            inertia = self._compute_inertia(X, centers, labels)

            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        """
        Predict cluster assignments for new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.cluster_centers_)
