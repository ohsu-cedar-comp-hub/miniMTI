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


def create_panel_selection_data(fpath1, fpath2, savename, n_clusters=15, n_closest=1000):
    f1 = h5py.File(fpath1)
    f2 = h5py.File(fpath2)
    
    mints1 = get_mints(f1)
    mints2 = get_mints(f2)
    
    mints = np.concatenate([mints1, mints2], axis=0)
    meta = np.concatenate([f1['metadata'][:], f2['metadata'][:]])
    ims = np.concatenate([f1['images'][:], f2['images'][:]], axis=0)
    masks = np.concatenate([f1['masks'][:], f2['masks'][:]], axis=0)
    
    
    kmeans = KMeans(n_clusters=n_clusters,max_iter=100000).fit(mints)
    closest_points = get_n_closest_to_centers(mints, kmeans, n_closest=n_closest)
    
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
    savename = 'lunaphore_panel_select_data'
    fpath1 = '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid=1010332.h5'
    fpath2 = '/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid=1010173.h5'
    
    create_panel_selection_data(fpath1, fpath2, savename)