"""
Image Clustering Dashboard - Interactive Visualization UI

This module provides a command-line interface for launching the Streamlit-based
clustering visualization dashboard. It allows specifying the data path containing
clustering results from the AI pipeline.

Usage:
    python dashboard.py --path_data /path/to/analysis/results
    
    Arguments:
        --path_data: Path to directory containing clustering results (required)

The dashboard provides:
- 3D visualization of clustered images
- Per-cluster analysis and filtering
- Clustering metrics visualization
- Silhouette score analysis across cluster counts
- Image preview and browsing capabilities
"""

import os
import sys
import argparse
import subprocess


def main():
    """
    Parse arguments and launch the Streamlit dashboard.
    
    Sets up the environment variable for the output path and launches
    the Streamlit application with the dashboard.
    """
    parser = argparse.ArgumentParser(
        description="Image Clustering Dashboard - Interactive visualization of clustering results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dashboard.py --path_data ./output
  python dashboard.py -d /path/to/clustering/results
  
The dashboard will be accessible at http://localhost:8501 by default.
        """
    )
    
    parser.add_argument(
        "--path_data", "-d",
        required=True,
        type=str,
        help="Path to directory containing clustering results (Excel/CSV files from pipeline)"
    )
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.isdir(args.path_data):
        print(f"Error: Data path does not exist: {args.path_data}")
        sys.exit(1)
    
    path_data = os.path.abspath(args.path_data)
    
    print("=" * 60)
    print("CLUSTERING DASHBOARD")
    print("=" * 60)
    print(f"Data path: {path_data}")
    print("=" * 60)
    print("\nStarting Streamlit dashboard...")
    print("The dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server.\n")
    
    # Set environment variable for the dashboard to use
    env = os.environ.copy()
    env['CLUSTERING_OUTPUT_PATH'] = path_data
    env['STREAMLIT_SERVER_HEADLESS'] = 'false'
    
    # Get the path to dashboard_clustering.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(current_dir, 'dashboard_clustering.py')
    
    if not os.path.exists(dashboard_script):
        print(f"Error: Dashboard script not found at {dashboard_script}")
        sys.exit(1)
    
    try:
        # Launch Streamlit with the dashboard
        subprocess.run(
            [sys.executable, '-m', 'streamlit', 'run', dashboard_script, '--logger.level=info'],
            env=env,
            check=False
        )
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
