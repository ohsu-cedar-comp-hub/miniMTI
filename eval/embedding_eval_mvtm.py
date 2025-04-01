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
    if emb_type = 'tokens':
        return model.tokenizer.quantize.embedding.weight.detach().cpu().numpy()
    if emb_type = 'markers':
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.savefig(f'plots/{model_id}/embedding_similarities.png')

def plot_tsne(embeddings, channel_labels, model_id):
    embedded = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(embedded[:, 0], embedded[:, 1], cmap='tab10')
    for i, label in enumerate(channel_labels):
        ax.annotate(label, (embedded[i, 0], embedded[i, 1]))
    plt.savefig(f'plots/{model_id}/tsne_plot.png')

def main():
    parser = argparse.ArgumentParser(description="Run MVTM analysis with checkpoint and parameters.")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters JSON file.')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on.')
    parser.add_argument('--model_id', type=str, default='crc-orion-if-1024', help='name for directory to save plots to')

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
    channel_labels, channel_idx, ch2idx = get_channel_info()
    

    # Get model embeddings
    embeddings = get_model_embeddings(model, params)

    # Plot results
    model_id = args.model_id
    plot_distance_matrix(embeddings, channel_labels, model_id)
    plot_tsne(embeddings, channel_labels, model_id)

if __name__ == '__main__':
    main()
