import os
import sys
import ast
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import rescale
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance_matrix
from eval_mae import IF_MAE
sys.path.append('../data')
#from process_cedar_biolib_immune import get_channel_info
from process_aced_immune import get_channel_info

def get_ckpt(ckpt_id):
    dir_ = f"../training/cedar-panel-reduction/{ckpt_id}/checkpoints/"
    fname = os.listdir(dir_)[0]
    return f"{dir_}/{fname}"

def get_model_embeddings(model, params):
    return model.mae.pos_embedding[0].detach().cpu().numpy()

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return idx, corr_array[idx, :][:, idx]

def plot_distance_matrix(embeddings, channel_labels, model_id):
    #idx, dists = cluster_corr(embeddings @ embeddings.T)
    idx, dists = cluster_corr(cosine_similarity(embeddings,embeddings))
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(dists, cmap='vlag', vmax=0.05, vmin=-0.05)
    reordered_labels = [channel_labels[i] for i in idx]
    ax.set_xticks(np.arange(len(channel_labels)) + 0.5, reordered_labels, fontsize=12)
    ax.set_yticks(np.arange(len(channel_labels)) + 0.5, reordered_labels, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.savefig(f'plots/{model_id}/embedding_similarities.png')
    
def plot_tsne(embeddings, channel_labels, model_id):
    embedded = TSNE(n_components=2,random_state=42, perplexity=5).fit_transform(embeddings)
    
    fig, ax = plt.subplots()
    ax.scatter(embedded[:,0], embedded[:,1], cmap='tab10')
    for i, label in enumerate(channel_labels):
        ax.annotate(label, (embedded[i,0], embedded[i,1]))
    plt.savefig(f'plots/{model_id}/tsne_plot.png')
    
if __name__ == '__main__':
    device = torch.device('cuda:1')
    model_id = 'djqmufd1'
    param_file="params.json"

    if not os.path.exists(f'plots/{model_id}'): os.mkdir(f'plots/{model_id}')

    with open(param_file) as f:
        params = json.load(f)

    ckpt = get_ckpt(model_id)
    model = IF_MAE(**params).load_from_checkpoint(ckpt, **params)
    model = model.to(device)
    model = model.eval()

    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    ch2stain = {i:ch for ch,i in ch2idx.items()}
    embeddings = get_model_embeddings(model, params)
    plot_distance_matrix(embeddings, ['cls'] + list(ch2stain.values()), model_id)
    plot_tsne(embeddings, ['cls'] + list(ch2stain.values()), model_id)