import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

from constant import PATH_OUTPUT

st.set_page_config(page_title="Cluster Buster Dashboard", layout="wide")

# -----------------------------
# Cached helpers
# -----------------------------
@st.cache_data
def load_clustering_df(filename: str) -> pd.DataFrame:
    path = os.path.join(PATH_OUTPUT, filename)
    return pd.read_excel(path)


@st.cache_data
def load_metric_df() -> pd.DataFrame:
    """
    Your updated pipeline writes only:
      - save_metric_resnet.xlsx (one-row dataframe)

    This loader keeps the door open for future descriptors by concatenating
    any metric files that exist.
    """
    metric_files = []

    # Current pipeline output:
    metric_files.append(os.path.join(PATH_OUTPUT, "save_metric_resnet.xlsx"))

    dfs = []
    for f in metric_files:
        if os.path.exists(f):
            df = pd.read_excel(f)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_metric = pd.concat(dfs, ignore_index=True)

    # Drop Excel index column if present
    if "Unnamed: 0" in df_metric.columns:
        df_metric = df_metric.drop(columns=["Unnamed: 0"])

    return df_metric


@st.cache_data
def plot_metric(df_metric: pd.DataFrame):
    if df_metric is None or df_metric.empty:
        return None

    # Find all numeric columns (metrics)
    metric_cols = [col for col in df_metric.columns if df_metric[col].dtype in ['float64', 'int64'] and col != 'descriptor']
    
    if not metric_cols:
        return None

    # Get the first (and only) row as we're showing 1 descriptor + 1 model
    row = df_metric.iloc[0]
    
    # Prepare data for radar chart
    radar_data = pd.DataFrame({
        'Metric': metric_cols,
        'Value': [row[col] for col in metric_cols]
    })
    
    # Create radar chart
    fig = px.bar_polar(
        radar_data,
        r='Value',
        theta='Metric',
        title="Clustering Metrics Profile",
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        height=500,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Assuming metrics are normalized 0-1
            )
        )
    )
    return fig


def _validate_cluster_df(df: pd.DataFrame) -> list[str]:
    required = {"x", "y", "z", "cluster"}
    missing = sorted(list(required - set(df.columns)))
    return missing


# -----------------------------
# Load data
# -----------------------------
st.title("Cluster Buster — Clustering Dashboard")

with st.sidebar:
    st.header("Settings")

    # Only RESNET is produced by your current pipeline snippet
    descriptor = st.selectbox("Sélectionner un descripteur", ["RESNET50"])

# Load selected descriptor clustering results
if descriptor == "RESNET50":
    clustering_filename = "save_clustering_resnet_kmeans.xlsx"
else:
    clustering_filename = None

df = load_clustering_df(clustering_filename)

# Clean excel index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

missing_cols = _validate_cluster_df(df)
if missing_cols:
    st.error(
        "Le fichier de clustering ne contient pas les colonnes requises pour la visualisation 3D.\n\n"
        f"Colonnes manquantes: {missing_cols}\n\n"
        f"Fichier chargé: {clustering_filename}"
    )
    st.stop()

# Ensure cluster is int-like for selector
try:
    clusters_sorted = sorted(pd.unique(df["cluster"]).tolist())
except Exception:
    clusters_sorted = list(range(10))

with st.sidebar:
    selected_cluster = st.selectbox("Sélectionner un Cluster", clusters_sorted)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global"])

# -----------------------------
# Tab 1: Per descriptor analysis
# -----------------------------
with tab1:
    st.subheader(f"Résultat de Clustering — Descripteur {descriptor}")
    st.caption(f"Source: {os.path.join(PATH_OUTPUT, clustering_filename)}")

    # 3D scatter
    st.write(f"#### Visualisation 3D du clustering avec descripteur {descriptor}")

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=df["cluster"].astype(str),
        title=f"Clustering avec descripteur {descriptor}",
    )
    fig.update_traces(marker_size=3)
    st.plotly_chart(fig, use_container_width=True)

    # Highlight selected cluster
    st.write(f"#### Focus sur le cluster {selected_cluster}")
    cluster_df = df[df["cluster"] == selected_cluster]
    fig2 = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=df["cluster"].astype(str),
        title=f"Cluster {selected_cluster} mis en évidence",
    )
    fig2.add_scatter3d(
        x=cluster_df["x"],
        y=cluster_df["y"],
        z=cluster_df["z"],
        mode="markers",
        marker=dict(color="red", size=6),
        name=f"Cluster {selected_cluster}",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Add label distribution chart
    st.write("#### Distribution des labels dans ce cluster")
    if 'label' in cluster_df.columns:
        label_counts = cluster_df['label'].value_counts()
        fig_dist = px.bar(
            x=label_counts.index,
            y=label_counts.values,
            labels={'x': 'Label', 'y': 'Nombre d\'éléments'},
            title=f"Distribution des labels - Cluster {selected_cluster}",
            color=label_counts.index
        )
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.write(f"**Total d'éléments dans ce cluster : {len(cluster_df)}**")


    # Display table
    st.write("#### Elements du cluster")
    st.dataframe(cluster_df.drop(columns=['image_path']), use_container_width=True, hide_index=False)
    
    # Pagination for images
    st.write("#### Visualisation des images par page")
    
    # Pagination settings
    images_per_page = st.selectbox("Images par page", [6, 9, 12, 15], index=1)
    
    total_images = len(cluster_df)
    total_pages = (total_images + images_per_page - 1) // images_per_page
    
    # Initialize session state for page number
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    # Pagination controls (centered)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button("← Previous"):
            st.session_state.current_page = max(0, st.session_state.current_page - 1)
    
    with col3:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
    
    with col4:
        if st.button("Next →"):
            st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
    
    # Calculate start and end indices for current page
    start_idx = st.session_state.current_page * images_per_page
    end_idx = min(start_idx + images_per_page, total_images)
    page_data = cluster_df.iloc[start_idx:end_idx]
    
    # Display images in a grid 
    cols = st.columns(4)
    
    for col_idx, (idx, row) in enumerate(page_data.iterrows()):
        col = cols[col_idx % 4]
        with col:
            try:
                image_path = row.get('image_path')
                label = row.get('label', 'unknown')
                
                # Fix path separators to be consistent on Windows
                if image_path:
                    image_path = image_path.replace('/', '\\')
                
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)                    
                    st.image(img, caption=f"{label} (ID: {idx})", use_column_width=True)
                else:
                    st.warning(f"Image not found")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.write(f"**Total d'éléments dans ce cluster : {len(cluster_df)}**")

# -----------------------------
# Tab 2: Global analysis (metrics)
# -----------------------------
with tab2:
    st.subheader("Analyse Global des descripteurs")

    df_metric = load_metric_df()
    if df_metric.empty:
        st.warning(
            "Aucun fichier de métriques trouvé. "
            "Assurez-vous que la pipeline a généré output/save_metric_resnet.xlsx."
        )
    else:
        fig_radar = plot_metric(df_metric)
        if fig_radar is not None:
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.write("## Métriques complètes")
        st.dataframe(df_metric, use_container_width=True)