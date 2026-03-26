# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies for image processing and other libraries
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Copy pre-computed clustering results (if available)
COPY output/ ./output/

EXPOSE 8501
EXPOSE 5000

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_LOGGER_LEVEL=info

# Default: Run the Streamlit dashboard
# To run pipeline: docker run -v /path/to/data:/data -v /path/to/output:/output myimage python src/pipeline.py --path_data /data --path_output /output
# To run dashboard with data: docker run -v /path/to/output:/app/output -p 8501:8501 myimage python src/dashboard.py --path_data /output
# Default dashboard: docker run -p 8501:8501 myimage
CMD ["streamlit", "run", "src/dashboard_clustering.py", "--logger.level=info"]

