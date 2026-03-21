from __future__ import annotations
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None, n_init=10, init="k-means++"):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.init = init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _init_centers_random(self, X, rng):
        n = X.shape[0]
        replace = n < self.n_clusters
        idx = rng.choice(n, size=self.n_clusters, replace=replace)
        return X[idx].copy()

    def _init_centers_kmeanspp(self, X, rng):
        n, d = X.shape
        centers = np.empty((self.n_clusters, d), dtype=X.dtype)

        # pick first center uniformly
        idx0 = rng.integers(0, n)
        centers[0] = X[idx0]

        # squared distance to closest center for each point
        closest_dist_sq = ((X - centers[0]) ** 2).sum(axis=1)

        for k in range(1, self.n_clusters):
            # probability proportional to distance^2
            probs = closest_dist_sq / np.sum(closest_dist_sq)
            idx = rng.choice(n, p=probs)
            centers[k] = X[idx]

            dist_sq_new_center = ((X - centers[k]) ** 2).sum(axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new_center)

        return centers

    def _init_centers(self, X, rng):
        if self.init == "random":
            return self._init_centers_random(X, rng)
        if self.init == "k-means++":
            return self._init_centers_kmeanspp(X, rng)
        raise ValueError("init must be 'random' or 'k-means++'")

    @staticmethod
    def _assign_labels(X, centers):
        # squared euclidean distances
        dist_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return dist_sq.argmin(axis=1)

    @staticmethod
    def _compute_centers(X, labels, n_clusters):
        d = X.shape[1]
        centers = np.empty((n_clusters, d), dtype=X.dtype)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = np.nan  # will be handled by re-seeding
        return centers

    @staticmethod
    def _compute_inertia(X, centers, labels):
        diff = X - centers[labels]
        return float(np.sum(diff * diff))

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        rng_master = self._rng()

        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for run in range(self.n_init):
            # different seed per run, reproducible if random_state provided
            seed = None if self.random_state is None else int(rng_master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(seed)

            centers = self._init_centers(X, rng)

            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centers)
                new_centers = self._compute_centers(X, labels, self.n_clusters)

                # handle empty clusters by re-seeding them to random points
                if np.isnan(new_centers).any():
                    empty = np.isnan(new_centers).any(axis=1)
                    idx = rng.choice(X.shape[0], size=int(empty.sum()), replace=False)
                    new_centers[empty] = X[idx]

                if np.allclose(new_centers, centers):
                    break
                centers = new_centers

            inertia = self._compute_inertia(X, centers, labels)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.cluster_centers_)




    

def show_metric(labels_true, labels_pred, descriptors,bool_return=False,name_descriptor="", name_model="kmeans",bool_show=True):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    # Affichons les résultats
    if bool_show :
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model}
