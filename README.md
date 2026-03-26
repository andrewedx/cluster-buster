# Image Clustering Pipeline - Snack Foods Classification

A comprehensive machine learning pipeline for image clustering and analysis. Extract multiple types of feature descriptors from images, apply various clustering algorithms, and visualize results through an interactive dashboard.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Run AI Pipeline](#2-run-ai-pipeline)
  - [3. Run Dashboard](#3-run-dashboard)
- [Docker Deployment](#docker-deployment)
- [Output Formats](#output-formats)
- [Project Structure](#project-structure)

## ✨ Features

### Feature Extractors
- **ResNet50**: Deep learning feature extraction
- **DINO-v2**: Vision transformer-based features
- **SIFT**: Scale-invariant feature transform
- **HOG**: Histogram of oriented gradients
- **GLCM**: Gray-level co-occurrence matrix texture features
- **Gray Histogram**: Intensity distribution analysis

### Clustering Algorithms
- **K-Means**: Partition-based clustering
- **Spectral Clustering**: Eigenvalue-based clustering
- **Gaussian Mixture Model (GMM)**: Probabilistic clustering
- **Agglomerative Clustering**: Hierarchical clustering

### Analysis & Metrics
- Silhouette score analysis across cluster counts
- Homogeneity, Completeness, V-measure scores
- Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
- Explained variance ratio for PCA
- 3D visualization of clustering results

### Dashboard
- Interactive 3D visualization of clustered features
- Per-cluster analysis and filtering
- Cluster composition analysis with label distribution
- Image preview and browsing capabilities
- Silhouette score sweep visualization

## 💻 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- CUDA GPU (optional, for faster processing)

### Step 1: Clone/Download the Project
```bash
cd /path/to/project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages including:
- scikit-learn for clustering algorithms
- TensorFlow and PyTorch for deep learning models
- Streamlit for the dashboard
- Pandas and NumPy for data processing
- Plotly for interactive visualizations

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run AI Pipeline

Execute the clustering pipeline on your image dataset:

```bash
python pipeline.py --path_data /path/to/images --path_output /path/to/output
```

**Arguments:**
- `--path_data` or `-d`: Path to directory containing images (required)
  - Images should be organized in subdirectories by class
  - Example structure: `images/data/class1/img1.jpg`, `images/data/class1/img2.jpg`, etc.
- `--path_output` or `-o`: Path to output directory for results (required)

**Example:**
```bash
python pipeline.py -d ./images/data/test -o ./output
```

**Output Files:**
The pipeline generates clustering results in both Excel and CSV formats:
- `save_clustering__<feature>__<model>.xlsx` - Clustering assignments and 3D coordinates
- `save_clustering__<feature>__<model>.csv` - Same as above in CSV format
- `save_metric__<feature>__<model>.xlsx` - Clustering metrics
- `save_metric__<feature>__<model>.csv` - Same as above in CSV format
- `sweep_silhouette__<feature>__<model>.json` - Silhouette scores for different cluster counts

### 3. Run Dashboard

Launch the interactive visualization dashboard:

```bash
python dashboard.py --path_data /path/to/output
```

**Arguments:**
- `--path_data` or `-d`: Path to directory containing clustering results (required)

**Example:**
```bash
python dashboard.py -d ./output
```

The dashboard will be accessible at `http://localhost:8501`

**Dashboard Features:**
- **Feature/Model Selection**: Choose different feature extractors and clustering algorithms from dropdowns
- **3D Visualization**: Interactive 3D scatter plot of clustered data points
- **Cluster Analysis**: 
  - Focus on specific clusters
  - View label distribution within clusters
  - Browse images in clusters (with pagination)
- **Metrics Tab**: View clustering quality metrics in radar/bar charts
- **Silhouette Analysis**: Visualize silhouette scores across different cluster numbers

### Complete Workflow Example

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Run the AI pipeline
python pipeline.py --path_data ./images/data/test --path_output ./results

# 3. Launch the dashboard
python dashboard.py --path_data ./results

# 4. Open browser to http://localhost:8501
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t clustering-pipeline:latest .
```

### Run Default Dashboard
```bash
docker run -p 8501:8501 clustering-pipeline:latest
```

This will start the Streamlit dashboard at `http://localhost:8501` using pre-computed results in the `/output` directory.



### Run Dashboard Container (with Pre-computed Results)
```bash
docker run -p 8501:8501 \
  -v /path/to/results:/app/output \
  clustering-pipeline:latest \
  python src/dashboard.py --path_data /app/output
```

### Run Pipeline in Container
```bash
docker run \
  -v /path/to/images:/data \
  -v /path/to/output:/output \
  clustering-pipeline:latest \
  python src/pipeline.py --path_data /data --path_output /output
```

## 📁 Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── DOCKER_INSTRUCTIONS.md       # Docker usage guide
│
├── src/                         # Source code (modular structure)
│   ├── __init__.py
│   ├── pipeline.py             # Main AI pipeline orchestrator
│   ├── dashboard.py            # Dashboard CLI wrapper
│   ├── dashboard_clustering.py # Streamlit dashboard application
│   │
│   ├── config/                 # Configuration module
│   │   ├── __init__.py
│   │   └── constant.py         # Path configuration constants
│   │
│   ├── features/               # Feature extraction module
│   │   ├── __init__.py
│   │   ├── neural.py           # ResNet50 & DINO-v2 extractors
│   │   ├── sift.py             # SIFT feature extraction
│   │   ├── glcm.py             # GLCM texture feature extraction
│   │   └── histogram.py        # Gray histogram & HOG features
│   │
│   ├── clustering/             # Clustering module
│   │   ├── __init__.py
│   │   ├── kmeans.py           # Custom K-Means implementation
│   │   ├── metrics.py          # Clustering evaluation metrics
│   │   └── sweep.py            # Silhouette score analysis
│   │
│   └── utils/                  # Utilities module
│       ├── __init__.py
│       ├── image.py            # Image loading & preprocessing
│       ├── visualization.py    # t-SNE dimensionality reduction
│       └── export.py           # DataFrame export functionality
│
├── images/                      # Image dataset
│   └── data/
│       ├── train/              # Training images
│       ├── val/                # Validation images
│       └── test/               # Test images (organized by class)
│
└── output/                      # Generated results (created by pipeline)
    ├── save_clustering__*.xlsx
    ├── save_clustering__*.csv
    ├── save_metric__*.xlsx
    ├── save_metric__*.csv
    └── sweep_silhouette__*.json
```

### Module Organization

**config/** - Centralized configuration
- Stores path constants (data, output directories)
- Reduces hardcoded paths across codebase

**features/** - Feature extraction (6 methods)
- `resnet50`: ImageNet pre-trained CNN (2048-dim)
- `dinov2`: Vision transformer embeddings (384-1536-dim)
- `sift`: Scale-invariant features + BoW (392-dim)
- `glcm`: Texture features (36-dim)
- `gray_histogram`: Intensity distribution (16-dim)
- `hog`: Histogram of gradients (~324-dim)

**clustering/** - Clustering algorithms & evaluation
- `kmeans.py`: Custom K-Means++ with multiple runs
- `metrics.py`: 6 evaluation metrics (Silhouette, ARI, AMI, etc.)
- `sweep.py`: Silhouette analysis across cluster counts

**utils/** - Common utilities
- `image.py`: EXIF-corrected image loading
- `visualization.py`: t-SNE 3D projection
- `export.py`: Excel/CSV data export

## 🔧 Advanced Usage

### Process Specific Features Only
Modify `FEATURES` list in `pipeline.py`:
```python
FEATURES = ["resnet50", "dinov2"]  # Only these features
```

### Process Specific Models Only
Modify `MODELS` list in `pipeline.py`:
```python
MODELS = ["kmeans", "spectral"]  # Only these clustering models
```

### Adjust PCA Components
In `pipeline.py`, modify the `pca_components` parameter:
```python
_run_one(
    # ... other parameters ...
    pca_components=128  # Increase for more detail, decrease for speedup
)
```

## 📈 Clustering Metrics Explained

- **Silhouette Score**: Measures how similar an object is to its cluster (-1 to 1, higher is better)
- **Homogeneity**: All samples of a single class belong to one cluster (0 to 1, higher is better)
- **Completeness**: All samples of a class are assigned to one cluster (0 to 1, higher is better)
- **V-measure**: Harmonic mean of homogeneity and completeness
- **Adjusted Rand Index (ARI)**: Similarity between true and predicted clusters (-1 to 1, higher is better)
- **Normalized Mutual Information (NMI)**: Mutual information between true and predicted labels (0 to 1, higher is better)

## 🛠️ Troubleshooting

### Memory Issues
If running out of memory:
1. Reduce the number of images in the input directory
2. Reduce `pca_components` value
3. Use GPU acceleration (install CUDA-enabled TensorFlow)

### Missing Module Errors
Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

### Image Not Found in Dashboard
Ensure image paths in output files are accessible from the dashboard location. Relative paths should be from the working directory when launching the dashboard.

### Dashboard Port Already in Use
Change the port:
```bash
streamlit run src/dashboard_clustering.py --server.port 8502
```

## 📝 References

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Visualization](https://plotly.com/python/)
- [OpenCV](https://opencv.org/)

## 📄 License

This project is part of an educational assignment for ET4 Info, Polytech Paris Saclay.
