import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import os
from PIL import Image, ImageOps

def conversion_3d(X, n_components=3,perplexity=50,random_state=42, early_exaggeration=10,n_iter=3000):
    """
    Conversion des vecteurs de N dimensions vers une dimension précise (n_components) pour la visualisation
    Input : X (array-like) : données à convertir en 3D
            n_components (int) : nombre de dimensions cibles (par défaut : 3)
            perplexity (float) : valeur de perplexité pour t-SNE (par défaut : 50)
            random_state (int) : graine pour la génération de nombres aléatoires (par défaut : 42)
            early_exaggeration (float) : facteur d'exagération pour t-SNE (par défaut : 10)
            n_iter (int) : nombre d'itérations pour t-SNE (par défaut : 3000)
    Output : X_3d (array-like) : données converties en 3D
    """
    tsne = TSNE(n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                max_iter=n_iter
               )
    X = np.array(X)
    X_3d = tsne.fit_transform(X)
    return X_3d


def create_df_to_export(data_3d, l_true_label,l_cluster):
    """
    Création d'un DataFrame pour stocker les données et les labels
    Input : data_3d (array-like) : données converties en 3D
            l_true_label (list) : liste des labels vrais
            l_cluster (list) : liste des labels de cluster
            l_path_img (list) : liste des chemins des images
    Output : df (DataFrame) : DataFrame contenant les données et les labels
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    
    return df

def image_loader(datasetPath):
    """
    Loads images and applies ONLY global preprocessing.
    Output is a list of Base Image dicts.
    """

    base_images = []
    labels = []

    for label in os.listdir(datasetPath):
        label_path = os.path.join(datasetPath, label)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            try:
                pil_img = Image.open(img_path)
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
            except:
                print(f"Error loading image: {img_path}")
                continue

            # Convert to float32 RGB in [0,1]
            img_np = np.asarray(pil_img).astype("float32") / 255.0

            base_img = {
                "data": img_np,                # H x W x 3 float32
                "width": img_np.shape[1],
                "height": img_np.shape[0],
                "path": img_path,
                "label_name": label
            }

            base_images.append(base_img)
            labels.append(label)

    return base_images, labels
