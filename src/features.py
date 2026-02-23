import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools



import matplotlib.pyplot as plt

from skimage import data, exposure

def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (array) : tableau numpy (n_samples, n_bins) des descripteurs d'histogrammes de niveau de gris
    """
    hist_list = []
    for image in images:
        # Convert image to uint8 format for cv2.calcHist
        image_uint8 = (image * 255 / 16).astype(np.uint8)
        hist = cv2.calcHist(
            [image_uint8],
            [0],
            None,
            [16],
            [0, 256]
        )
        # normalize the histogram
        hist = hist / np.sum(hist)
        # flatten to 1D
        hist = hist.flatten().astype(float)
        hist_list.append(hist)
    return np.vstack(hist_list)

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (array) : tableau numpy (n_samples, n_features) des descripteurs HOG
    """
    fd_list = []
    for image in images:
        fd = hog(
            image,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            visualize=False,
            feature_vector=True
        )
        fd_list.append(fd.astype(float))
    return np.vstack(fd_list)
    

