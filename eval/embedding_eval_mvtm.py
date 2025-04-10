import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('../training/MVTM')
from mvtm import IF_MVTM

def get_model_embeddings(model, params, emb_type='markers'):
    if emb_type == 'tokens':
        return model.tokenizer.quantize.embedding.weight.detach().cpu().numpy()
    if emb_type == 'markers':
        return model.mvtm.roberta.embeddings.token_type_embeddings.weight.cpu().detach().numpy()

def cluster_corr(corr_array, inplace=False):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return idx, corr_array[idx, :][:, idx]

def plot_distance_matrix(embeddings, channel_labels, model_id):
    idx, dists = cluster_corr(cosine_similarity(embeddings, embeddings))
    fig, ax = plt.subplots(figsize=(embeddings.shape[0], embeddings.shape[0]))
    sns.heatmap(dists, cmap='vlag', vmax=0.05, vmin=-0.05)
    
    # Create position ticks for labels
    positions = np.arange(len(idx)) + 0.5
    ordered_labels = [channel_labels[i] for i in idx]
    
    # Set the tick positions and labels
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(ordered_labels)
    ax.set_yticklabels(ordered_labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.savefig(f'plots/{model_id}/embedding_similarities.png')

def plot_tsne(embeddings, channel_labels, model_id, n_clusters=5):
    # Apply KMeans clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Apply t-SNE for dimensionality reduction
    embedded = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=clusters, cmap='tab10', s=100)
    
    # Add channel labels
    for i, label in enumerate(channel_labels):
        ax.annotate(label, (embedded[i, 0], embedded[i, 1]), 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Add legend for clusters
    legend = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    ax.add_artist(legend)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'plots/{model_id}/tsne_plot.png', dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Run MVTM analysis with checkpoint and parameters.")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters JSON file.')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on.')
    parser.add_argument('--model_id', type=str, default='crc-orion-if-1024', help='name for directory to save plots to')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for K-means in t-SNE plot')

    args = parser.parse_args()

    # Load parameters
    with open(args.params) as f:
        params = json.load(f)

    # Load model
    model = IF_MVTM(**params).load_from_checkpoint(args.ckpt, **params)
    model = model.to(args.device)
    model = model.eval()

    # Define channel-to-index mapping
    sys.path.append('../data')
    from lunaphore_channel_info import get_channel_info
    #from crc_orion_channel_info import get_channel_info
    channel_labels, channel_idx, ch2idx = get_channel_info()
    

    # Get model embeddings
    embeddings = get_model_embeddings(model, params)

    # Create output directory if it doesn't exist
    os.makedirs(f'plots/{args.model_id}', exist_ok=True)
    
    # Plot results
    model_id = args.model_id
    plot_distance_matrix(embeddings, list(ch2idx.keys()), model_id)
    plot_tsne(embeddings, list(ch2idx.keys()), model_id, n_clusters=args.n_clusters)

if __name__ == '__main__':
    main()
