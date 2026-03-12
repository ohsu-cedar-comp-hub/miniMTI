import h5py
import numpy as np
from einops import repeat
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

def get_mints(h5):
    masks = h5['masks'][:]
    ims = h5['images'][:]
    masks = repeat(masks,'b h w -> b h w c', c=ims.shape[-1])
    mints = (ims * masks).sum(axis=(1,2)) / masks.sum(axis=(1,2))
    return mints

def get_n_closest_to_centers(data, kmeans, n_closest=5):
    # Get distances from each point to each center
    distances = euclidean_distances(data, kmeans.cluster_centers_)
    
    # For each center, get the n closest points
    closest_indices = {}
    for cluster_idx in range(len(kmeans.cluster_centers_)):
        # Get distances to this center
        cluster_distances = distances[:, cluster_idx]
        
        # Find indices of n smallest distances
        closest_n = np.argsort(cluster_distances)[:n_closest]
        
        # Store in dictionary with cluster number as key
        closest_indices[cluster_idx] = closest_n
        
    return closest_indices


def get_n_random_points_in_clusters(data, kmeans, n_samples=5):
    """
    Samples n random points from each cluster.

    Args:
        data: The input data (NumPy array).
        kmeans: The fitted KMeans model.
        n_samples: The number of random samples to draw from each cluster.

    Returns:
        A dictionary where keys are cluster indices and values are lists of
        indices of the randomly sampled points within that cluster.  Also returns
        a dictionary containing the distances from the samples to their centroid.
    """

    distances = euclidean_distances(data, kmeans.cluster_centers_)
    
    cluster_assignments = kmeans.labels_
    sampled_indices = {}
    
    for cluster_idx in range(kmeans.cluster_centers_.shape[0]):
        # Find indices of points belonging to the current cluster
        points_in_cluster = np.where(cluster_assignments == cluster_idx)[0]

        # Handle the case where a cluster has fewer than n_samples points
        num_to_sample = min(n_samples, len(points_in_cluster))

        # Randomly sample n_samples indices from the points in the cluster
        if num_to_sample > 0:  # Avoid errors with empty clusters
           sampled_indices[cluster_idx] = np.random.choice(
                points_in_cluster, size=num_to_sample, replace=False
            )
        else:
           sampled_indices[cluster_idx] = np.array([]) # Or some other indicator of an empty sample

    return sampled_indices


def create_panel_selection_data(fpath1, fpath2, savename, n_clusters=15, n_closest=100):
    f1 = h5py.File(fpath1)
    mints1 = get_mints(f1)
    
    if fpath2 is not None:
        f2 = h5py.File(fpath2)
        mints2 = get_mints(f2)  
        mints = np.concatenate([mints1, mints2], axis=0)
        meta = np.concatenate([f1['metadata'][:], f2['metadata'][:]])
        ims = np.concatenate([f1['images'][:], f2['images'][:]], axis=0)
        masks = np.concatenate([f1['masks'][:], f2['masks'][:]], axis=0)
    else:
        mints = mints1
        meta = f1['metadata'][:]
        ims = f1['images'][:]
        masks = f1['masks'][:]  
        
    
    
    kmeans = KMeans(n_clusters=n_clusters,max_iter=100000).fit(mints)
    #closest_points = get_n_closest_to_centers(mints, kmeans, n_closest=n_closest)
    closest_points = get_n_random_points_in_clusters(mints, kmeans, n_samples=n_closest)
    
    rep_images = np.concatenate([ims[closest_points[c]] for c in closest_points.keys()], axis=0)
    rep_masks = np.concatenate([masks[closest_points[c]] for c in closest_points.keys()], axis=0)
    rep_meta = np.concatenate([meta[closest_points[c]] for c in closest_points.keys()], axis=0)
    
    save_path = '/'.join(fpath1.split('/')[:-1])
    save_path += f'/{savename}.h5'
    with h5py.File(save_path, 'w') as f:
        images = f.create_dataset('images',data=rep_images)
        masks = f.create_dataset('masks',data=rep_masks)
        metas = f.create_dataset('metadata',data=rep_meta)


if __name__ == '__main__':
    #savename = 'lunaphore_panel_select_data'
    #fpath1 = '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid=1010332.h5'
    #fpath2 = '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid=1010173.h5'
    #fpath1 = '/arc/scratch1/ChangLab/lunaphore-immune-unnorm/lunaphore_dataset_unnorm_sid=1010240.h5'
    #fpath2 = '/arc/scratch1/ChangLab/lunaphore-immune-unnorm/lunaphore_dataset_unnorm_sid=1010296.h5'
    #fpath3 = '/arc/scratch1/ChangLab/lunaphore-immune-unnorm/lunaphore_dataset_unnorm_sid=1010323.h5'
    
    savename = 'orion_panel_select_data_CRC02_CRC03'
    fpath1 = '/home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All/orion_crc_dataset_sid=CRC02.h5'
    fpath2 = '/home/exacloud/gscratch/ChangLab/ORION-CRC-Unnormalized-All/orion_crc_dataset_sid=CRC03.h5'
    
    
    create_panel_selection_data(fpath1, fpath2, savename)