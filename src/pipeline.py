from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.decomposition import PCA
import os
import pandas as pd
from sklearn import datasets

from features import *
from clustering import *
from resnet import compute_dinov2_descriptors, compute_resnet50_descriptors
from utils import *
from constant import *



def pipeline():
    print("##########   LOADING IMAGES  ##########")
    base_images, labels_true = image_loader(IMAGES_DIR)
    print(f"loaded {len(base_images)} images with {len(set(labels_true))} classes")
    
    # Diagnostic: class distribution
    from collections import Counter
    class_counts = Counter(labels_true)
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} images")
    
    # Convert string labels to integers for compatibility with clustering metrics
    label_encoder = LabelEncoder()
    labels_true_encoded = label_encoder.fit_transform(labels_true)
    print(f"Labels encoded from {len(set(labels_true))} unique string labels to integers")

    print("\n\n ##### Extraction de Features ######")
    print("- calcul features ResNet50...")
    # descriptors_resnet = compute_resnet50_descriptors(base_images)
    descriptors_dino = compute_dinov2_descriptors(base_images)
    # print(f"descriptors_resnet shape: {descriptors_resnet.shape}")
    print(f"descriptors_dino shape: {descriptors_dino.shape}")
    
    print("- PCA dimensionality reduction (256 components)...")
    pca = PCA(n_components=32, whiten=False, random_state=42)
    descriptors_resnet_pca = pca.fit_transform(descriptors_dino)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    print("- L2 normalization...")
    descriptors_resnet_norm = normalize(descriptors_resnet_pca, norm='l2')
    print(f"descriptors_resnet_norm shape: {descriptors_resnet_norm.shape}")


    print("\n\n ##### Clustering ######")
    number_cluster = len(set(labels_true_encoded))

    print(f"- Running k-means with {number_cluster} clusters (n_init=20)...")
    kmeans_resnet = KMeans(n_clusters=number_cluster, max_iter=300, n_init=20, random_state=42)
    kmeans_resnet.fit(descriptors_resnet_norm)

    print("\n\n ##### Résultat ######")
    metric_resnet = show_metric(
        labels_true_encoded,
        kmeans_resnet.labels_,
        descriptors_resnet_norm,
        bool_show=True,
        name_descriptor="RESNET50",
        bool_return=True
    )
    
    print("- réduction en 3D pour visualisation...")
    x_3d_resnet = conversion_3d(descriptors_resnet_norm)

    df_resnet = create_df_to_export(
        x_3d_resnet,
        labels_true,
        kmeans_resnet.labels_,
        base_images
    )

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_resnet.to_excel(PATH_OUTPUT + "/save_clustering_resnet_kmeans.xlsx")
    pd.DataFrame([metric_resnet]).to_excel(PATH_OUTPUT + "/save_metric_resnet.xlsx")

    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer : streamlit run dashboard_clustering.py")



def pipeline_old():
   
    digits = datasets.load_digits()


    labels_true =digits.target
    images = digits.images
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    # TODO
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    # TODO
    descriptors_hist = compute_gray_histograms(images)


    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)


    print("\n\n ##### Clustering ######")
    number_cluster = 10
    print("- calcul kmeans avec features HOG ...")
    # TODO
    kmeans_hist = KMeans(n_clusters=number_cluster, max_iter=100)

    print("- calcul kmeans avec features Histogram...")
    # TODO
    kmeans_hog = KMeans(n_clusters=number_cluster, max_iter=100)

    kmeans_hist.fit(descriptors_hist)
    kmeans_hog.fit(descriptors_hog)

    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True)


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist,metric_hog]
    df_metric = pd.DataFrame(list_dict)
    
    

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()