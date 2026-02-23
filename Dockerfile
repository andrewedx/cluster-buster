# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requierements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requierements.txt

# Copy the source code
COPY src/ ./src/

# Copy the pre-computed clustering output results
COPY output/ ./output/

# Expose the Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_LOGGER_LEVEL=info

# Run the Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard_clustering.py"]
