import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_OUTPUT = os.path.join(ROOT_DIR, "output")
IMAGES_DIR = os.path.join(ROOT_DIR, "images/data/test")
MODEL_CLUSTERING = "kmeans"