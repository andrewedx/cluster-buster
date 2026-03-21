from __future__ import annotations

import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

from features import *
from clustering import *
from resnet import compute_dinov2_descriptors, compute_resnet50_descriptors
from utils import *
from constant import *


FEATURES = ["resnet50", "dinov2", "gray_histogram", "hog"]  # add new features here
MODELS = ["kmeans", "spectral"]  # add new clustering models here


def _make_output_filenames(feature: str, model: str) -> tuple[str, str]:
    feature_key = feature.lower()
    model_key = model.lower()
    clustering_filename = f"save_clustering__{feature_key}__{model_key}.xlsx"
    metric_filename = f"save_metric__{feature_key}__{model_key}.xlsx"
    return clustering_filename, metric_filename


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


def _preprocess_descriptors(feature: str, descriptors, pca_components: int):
    """
    Feature-specific preprocessing rules:
    - gray_histogram: skip PCA by default (very low-dim), do L2 on raw hist
    - hog: StandardScaler helps, PCA optional but must be safe
    - dinov2/resnet50: PCA + L2 is recommended
    """
    X = descriptors

    # Skip PCA for HOG, but do StandardScaler
    if feature == "hog":
        X = StandardScaler().fit_transform(X)

    # Skip PCA for histogram
    if feature == "gray_histogram":
        X_pca = X
        pca = None
    else:
        X_pca, pca = _safe_pca_transform(X, pca_components)

    X_norm = normalize(X_pca, norm="l2")
    return X_norm, pca


def _run_one(
    *,
    feature: str,
    model: str,
    base_images: list[dict],
    labels_true: list[str],
    labels_true_encoded,
    pca_components: int = 64,
):
    # -------- Feature extraction --------
    if feature == "resnet50":
        descriptors = compute_resnet50_descriptors(base_images)
    elif feature == "dinov2":
        descriptors = compute_dinov2_descriptors(base_images)
    elif feature == "gray_histogram":
        descriptors = compute_gray_histograms_base_images(base_images, n_bins=16)
    elif feature == "hog":
        descriptors = compute_hog_descriptors_base_images(base_images)
    else:
        raise ValueError(f"Unknown feature: {feature}")

    descriptors = np.asarray(descriptors, dtype=np.float32)
    print(f"[{feature}/{model}] descriptors shape: {descriptors.shape}")

    # -------- Preprocess: (optional scaling) + (safe PCA) + L2 --------
    descriptors_norm, pca = _preprocess_descriptors(feature, descriptors, pca_components)

    if pca is not None:
        explained = float(pca.explained_variance_ratio_.sum())
        used_components = int(pca.n_components_)
        print(f"[{feature}/{model}] PCA components used: {used_components} (requested {pca_components})")
        print(f"[{feature}/{model}] explained variance sum: {explained:.4f}")
    else:
        explained = None
        used_components = None
        print(f"[{feature}/{model}] PCA skipped")

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
        # SpectralClustering returns labels directly (no .fit_predict stored labels_)
        # Note: n_neighbors must be < n_samples
        n_neighbors = min(20, descriptors_norm.shape[0] - 1)

        clusterer = SpectralClustering(
            n_clusters=number_cluster,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=42,
        )
        labels_pred = clusterer.fit_predict(descriptors_norm)

    else:
        raise ValueError(f"Unknown model: {model}")

    # -------- Metrics --------
    metric = show_metric(
        labels_true_encoded,
        labels_pred,
        descriptors_norm,
        bool_show=True,
        name_descriptor=feature.upper(),
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

    clustering_filename, metric_filename = _make_output_filenames(feature, model)

    os.makedirs(PATH_OUTPUT, exist_ok=True)
    df_cluster.to_excel(os.path.join(PATH_OUTPUT, clustering_filename), index=False)
    pd.DataFrame([metric]).to_excel(os.path.join(PATH_OUTPUT, metric_filename), index=False)

    print(f"[{feature}/{model}] wrote: {clustering_filename}")
    print(f"[{feature}/{model}] wrote: {metric_filename}")


def pipeline():
    print("##########   LOADING IMAGES  ##########")
    base_images, labels_true = image_loader(IMAGES_DIR)
    print(f"loaded {len(base_images)} images with {len(set(labels_true))} classes")

    # Convert string labels to integers for compatibility with clustering metrics
    label_encoder = LabelEncoder()
    labels_true_encoded = label_encoder.fit_transform(labels_true)

    # Run all feature/model combinations
    for feature in FEATURES:
        for model in MODELS:
            _run_one(
                feature=feature,
                model=model,
                base_images=base_images,
                labels_true=labels_true,
                labels_true_encoded=labels_true_encoded,
                pca_components=32,
            )

    print("Fin.\n\nPour avoir la visualisation dashboard, veuillez lancer : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()