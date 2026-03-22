import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def compute_sift_descriptors(base_images, vocab_size=200, img_size=(128, 128)):
    """
    Calcule les descripteurs SIFT + histogramme couleur RGB pour les images Snack.
    La couleur aide énormément à distinguer les fruits (rouge/jaune/vert/orange...)

    Input :
    - base_images (list of dict) : liste de dicts avec clé 'data' (H x W x 3 float32 [0,1])
    - vocab_size (int) : nombre de mots visuels BoW
    - img_size (tuple) : taille cible pour le resize

    Output :
    - final_features (np.array) : tableau (n_samples, vocab_size + color_bins)
    """
    sift = cv2.SIFT_create()
    all_raw_descriptors = []
    per_image_descriptors = []
    color_features_list = []

    print(f"  [SIFT] Extraction sur {len(base_images)} images...")
    print(f"  [SIFT] Resize → {img_size} | vocab_size → {vocab_size}")

    # ── PASS 1 : extraire SIFT + couleur pour chaque image ───────────────────
    for base_img in base_images:
        img_np = base_img["data"]  # float32 [0,1], H x W x 3 RGB

        # Convertir en uint8
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Resize à taille fixe
        img_resized = cv2.resize(img_uint8, img_size)

        # ── SIFT sur grayscale ────────────────────────────────────────────────
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        _, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            per_image_descriptors.append(descriptors)
            all_raw_descriptors.append(descriptors)
        else:
            per_image_descriptors.append(None)

        # ── Histogramme couleur RGB (32 bins par canal) ───────────────────────
        # 🍎 rouge  → canal R fort
        # 🍌 jaune  → canal R + G forts
        # 🥝 vert   → canal G fort
        # 🍊 orange → canal R fort + G moyen
        color_hist = []
        for canal in range(3):  # R, G, B
            hist, _ = np.histogram(
                img_resized[:, :, canal],
                bins=32,
                range=(0, 256)
            )
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # normalisation
            color_hist.extend(hist)

        color_features_list.append(np.array(color_hist))  # vecteur de 96 valeurs (32x3)

    # ── PASS 2 : construire le vocabulaire visuel SIFT ───────────────────────
    print(f"  [SIFT] Construction du vocabulaire visuel ({vocab_size} mots)...")
    stacked = np.vstack(all_raw_descriptors)
    print(f"  [SIFT] Total descripteurs bruts : {stacked.shape}")

    kmeans_vocab = MiniBatchKMeans(
        n_clusters=vocab_size,
        random_state=42,
        batch_size=2048,
        n_init=5
    )
    kmeans_vocab.fit(stacked)

    # ── PASS 3 : encoder chaque image en histogramme BoW ─────────────────────
    bow_features = []
    for descriptors in per_image_descriptors:
        if descriptors is not None:
            word_ids = kmeans_vocab.predict(descriptors)
            hist, _ = np.histogram(word_ids, bins=np.arange(vocab_size + 1))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
        else:
            hist = np.zeros(vocab_size)
        bow_features.append(hist)

    bow_features = np.vstack(bow_features)           # (952, vocab_size)
    color_features = np.vstack(color_features_list)  # (952, 96)

    # ── PASS 4 : combiner SIFT + couleur ─────────────────────────────────────
    # On donne plus de poids à la couleur car c'est très discriminant pour les fruits
    color_weight = 2.0   # 🎨 la couleur compte double
    final_features = np.hstack([
        bow_features,
        color_features * color_weight
    ])

    print(f"  [SIFT] SIFT BoW shape   : {bow_features.shape}")
    print(f"  [SIFT] Couleur shape    : {color_features.shape}")
    print(f"  [SIFT] Features finale  : {final_features.shape}")

    return final_features