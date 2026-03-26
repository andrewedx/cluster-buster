"""
Image Clustering Pipeline - Modular package structure.

Main entry points:
- pipeline.py: Run the AI clustering pipeline
- dashboard.py: Launch the interactive visualization dashboard
- dashboard_clustering.py: Streamlit UI (invoked by dashboard.py)
"""

# Note: Most imports are done directly in pipeline.py and dashboard.py
# to avoid circular dependencies. Users should run:
#   python pipeline.py --path_data <path> --path_output <path>
#   python dashboard.py --path_data <path>
