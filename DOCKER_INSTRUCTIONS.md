# Docker Setup Instructions

## Overview
This Dockerfile creates a containerized version of the SNACK image clustering Streamlit dashboard. The container runs the pre-built dashboard without training.

## Prerequisites
- Docker installed on your system
- Pre-computed clustering results in the `output/` directory (run `pipeline.py` first if needed)

## Building the Docker Image

```bash
docker build -t snack-clustering-dashboard .
```

This command:
- Tags the image as `snack-clustering-dashboard`
- Packages your application with all dependencies
- Sets up the Streamlit environment

## Running the Docker Container

### Basic usage:
```bash
docker run -p 8501:8501 snack-clustering-dashboard
```

### Access the dashboard:
Open your browser and navigate to:
```
http://localhost:8501
```

### Advanced options:

**Run in detached mode (background):**
```bash
docker run -d -p 8501:8501 --name clustering-app snack-clustering-dashboard
```

**View logs:**
```bash
docker logs clustering-app
```

**Stop the container:**
```bash
docker stop clustering-app
```

## What's Included
- Python 3.10 slim base image
- All dependencies from requirements.txt
- Source code from the `src/` directory
- Pre-computed clustering results from the `output/` directory

## Notes
- The container does NOT run the training pipeline (pipeline.py)
- It only serves the Streamlit dashboard
- Pre-computed results must be present in the `output/` directory
- The dashboard is accessible on port 8501

## Rebuilding
If you modify the source code or install new dependencies, rebuild the image:
```bash
docker build -t snack-clustering-dashboard .
```
