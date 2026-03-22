import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

from constant import PATH_OUTPUT

st.set_page_config(page_title="Cluster Buster Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _make_output_filenames(feature: str, model: str) -> tuple[str, str]:
    """
    Convention:
      save_clustering__<feature>__<model>.xlsx
      save_metric__<feature>__<model>.xlsx
    """
    feature_key = feature.lower()
    model_key = model.lower()
    clustering_filename = f"save_clustering__{feature_key}__{model_key}.xlsx"
    metric_filename = f"save_metric__{feature_key}__{model_key}.xlsx"
    return clustering_filename, metric_filename


@st.cache_data
def load_excel_df(filename: str) -> pd.DataFrame:
    path = os.path.join(PATH_OUTPUT, filename)
    return pd.read_excel(path)


def _drop_excel_index_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and "Unnamed: 0" in df.columns:
        return df.drop(columns=["Unnamed: 0"])
    return df


def _validate_cluster_df(df: pd.DataFrame) -> list[str]:
    required = {"x", "y", "z", "cluster"}
    missing = sorted(list(required - set(df.columns)))
    return missing


@st.cache_data
def plot_metric(metric_df: pd.DataFrame):
    """
    Expects a 1-row dataframe with metric columns.
    """
    if metric_df is None or metric_df.empty:
        return None

    metric_df = _drop_excel_index_col(metric_df)

    # Keep only numeric metrics
    metric_cols = [
        c for c in metric_df.columns
        if c not in ("descriptor", "feature", "model")
        and pd.api.types.is_numeric_dtype(metric_df[c])
    ]
    if not metric_cols:
        return None

    row = metric_df.iloc[0]
    radar_data = pd.DataFrame({"Metric": metric_cols, "Value": [float(row[c]) for c in metric_cols]})

    fig = px.bar_polar(
        radar_data,
        r="Value",
        theta="Metric",
        title="Clustering Metrics Profile",
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(
        height=500,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Cluster Buster — Clustering Dashboard")

with st.sidebar:
    st.header("Settings")

    # Dropdown 1: Feature extractor
    feature = st.selectbox("Feature", ["RESNET50", "DINOV2", "GRAY_HISTOGRAM", "HOG", "SIFT"])

    # Dropdown 2: Clustering model
    model = st.selectbox("Clustering model", ["KMEANS", "SPECTRAL", "GMM_FULL", "GMM_DIAG"])

    clustering_filename, metric_filename = _make_output_filenames(feature, model)

st.caption(f"Output folder: {PATH_OUTPUT}")
st.caption(f"Selected clustering file: {clustering_filename}")
st.caption(f"Selected metric file: {metric_filename}")

# -----------------------------
# Load clustering file
# -----------------------------
clustering_path = os.path.join(PATH_OUTPUT, clustering_filename)
if not os.path.exists(clustering_path):
    st.error(
        "Clustering file not found.\n\n"
        f"Expected: {clustering_path}\n\n"
        "Run the pipeline for this (feature, model) combination to generate it."
    )
    st.stop()

df = load_excel_df(clustering_filename)
df = _drop_excel_index_col(df)

missing_cols = _validate_cluster_df(df)
if missing_cols:
    st.error(
        "Clustering file is missing required columns for 3D visualization.\n\n"
        f"Missing columns: {missing_cols}\n"
        f"File: {clustering_filename}"
    )
    st.stop()

clusters_sorted = sorted(pd.unique(df["cluster"]).tolist())

with st.sidebar:
    selected_cluster = st.selectbox("Cluster", clusters_sorted)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Analyse par cluster", "Analyse globale"])

# -----------------------------
# Tab 1: Cluster view
# -----------------------------
with tab1:
    st.subheader(f"Clustering — Feature={feature}, Model={model}")

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=df["cluster"].astype(str),
        title=f"3D view — {feature} + {model}",
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

    st.write(f"### Focus cluster {selected_cluster}")
    cluster_df = df[df["cluster"] == selected_cluster]

    # Label distribution if present
    if "label" in cluster_df.columns:
        label_counts = cluster_df["label"].value_counts()
        fig_dist = px.bar(
            x=label_counts.index,
            y=label_counts.values,
            labels={"x": "Label", "y": "Count"},
            title=f"Label distribution — cluster {selected_cluster}",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.write(f"Elements in cluster: {len(cluster_df)}")
    st.dataframe(cluster_df, use_container_width=True, hide_index=False)

    # Images if image_path exists
    if "image_path" in cluster_df.columns:
        st.write("### Images (if available)")

        images_per_page = st.selectbox("Images per page", [6, 9, 12, 15], index=1)
        total_images = len(cluster_df)
        total_pages = (total_images + images_per_page - 1) // images_per_page

        if "current_page" not in st.session_state:
            st.session_state.current_page = 0

        col_prev, col_mid, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("← Previous"):
                st.session_state.current_page = max(0, st.session_state.current_page - 1)
        with col_mid:
            st.write(f"Page {st.session_state.current_page + 1} / {max(total_pages,1)}")
        with col_next:
            if st.button("Next →"):
                st.session_state.current_page = min(max(total_pages - 1, 0), st.session_state.current_page + 1)

        start_idx = st.session_state.current_page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        page_data = cluster_df.iloc[start_idx:end_idx]

        cols = st.columns(4)
        for col_idx, (_, row) in enumerate(page_data.iterrows()):
            col = cols[col_idx % 4]
            with col:
                image_path = row.get("image_path")
                label = row.get("label", "unknown")

                # cross-platform normalization
                if isinstance(image_path, str):
                    image_path = os.path.normpath(image_path)

                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    st.image(img, caption=str(label), use_column_width=True)
                else:
                    st.caption("Image not found")

# -----------------------------
# Tab 2: Global metrics
# -----------------------------
with tab2:
    st.subheader(f"Metrics — Feature={feature}, Model={model}")

    metric_path = os.path.join(PATH_OUTPUT, metric_filename)
    if not os.path.exists(metric_path):
        st.warning(
            "Metric file not found.\n\n"
            f"Expected: {metric_path}\n\n"
            "Run the pipeline for this (feature, model) combination to generate it."
        )
    else:
        df_metric = load_excel_df(metric_filename)
        df_metric = _drop_excel_index_col(df_metric)

        fig_radar = plot_metric(df_metric)
        if fig_radar is not None:
            st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(df_metric, use_container_width=True)