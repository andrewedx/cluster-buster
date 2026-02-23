import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from constant import PATH_OUTPUT
import os

# Load images dataset once
@st.cache_data
def load_images_data():
    digits = load_digits()
    return digits.images, digits.target


@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    fig = px.bar(df_metric, x='descriptor', y='ami', 
                 title='Comparison of AMI Scores by Descriptor',
                 labels={'descriptor': 'Descriptor Type', 'ami': 'Adjusted Mutual Information Score'},
                 color='descriptor',
                 color_discrete_map={'HISTOGRAM': '#636EFA', 'HOG': '#EF553B'})
    fig.update_layout(height=500, showlegend=False)
    return fig

        
# Chargement des données du clustering
df_hist = pd.read_excel(os.path.join(PATH_OUTPUT, "save_clustering_hist_kmeans.xlsx"))
df_hog = pd.read_excel(os.path.join(PATH_OUTPUT, "save_clustering_hog_kmeans.xlsx"))
df_metric = pd.read_excel(os.path.join(PATH_OUTPUT, "save_metric.xlsx"))

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données DIGITS')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', ["HISTOGRAM","HOG"])
    if descriptor=="HISTOGRAM":
        df = df_hist
    if descriptor=="HOG":
        df = df_hog
    # Ajouter un sélecteur pour les clusters
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', range(10))
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    # à remplir
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['cluster'].astype(str), title=f'Clustering avec descripteur {descriptor}')
    fig.update_traces(marker_size=3)
    st.plotly_chart(fig)
    
    # Display example images from selected cluster
    st.write(f"#### Exemples d'images du cluster {selected_cluster}")
    images, true_labels = load_images_data()
    
    # Get images corresponding to selected cluster
    cluster_image_indices = cluster_indices.tolist()
    cluster_images = images[cluster_image_indices]
    cluster_true_labels = true_labels[cluster_image_indices]
    
    # Display up to 15 example images in a grid
    n_examples = min(15, len(cluster_images))
    cols = st.columns(5)
    
    for i in range(n_examples):
        col = cols[i % 5]
        with col:
            fig_img = plt.figure(figsize=(2, 2))
            plt.imshow(cluster_images[i], cmap='gray')
            plt.axis('off')
            # Add true digit label as title
            plt.title(f"Digit: {cluster_true_labels[i]}", fontsize=10)
            st.pyplot(fig_img, use_container_width=True)
    
    st.write(f"**Total d'images dans ce cluster : {len(cluster_images)}**")

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    fig_ami = plot_metric(df_metric)
    st.plotly_chart(fig_ami, use_container_width=True)
    
    st.write('## Métriques complètes' )
    st.dataframe(df_metric, use_container_width=True)
