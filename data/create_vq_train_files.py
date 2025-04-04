from tqdm import tqdm
import h5py
from einops import rearrange
import numpy as np
from skimage.io import imsave
import histomicstk as htk
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.preprocessing.color_conversion import lab_mean_std

def deconvolve(he_im):
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # set to null if input contains only two stains
    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(he_im, W)
    he_im = imDeconvolved.Stains[:,:,:2]
    return he_im

savedir = '/arc/scratch1/ChangLab/lunaphore-vq-if-he-train-images'
data = h5py.File(f'/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized-All/lunaphore_vq_if_he_train_data.h5')
num_channels = 43
for i,im in tqdm(enumerate(data['images'])):
    for c in range(num_channels):
        np.save(f'{savedir}/sample-{i}-ch-{c}.npy', im[:,:,c])
    he_im = im[:,:,-3:]
    deconvolved_im = deconvolve(he_im)
    for c in range(2):
        np.save(f'{savedir}/sample-{i}-he-ch-{c}.npy', deconvolved_im[:,:,c])
    #imsave(f'{savedir}/{sid}-sample-{i}-HE.tif', he_im.astype('uint8'), check_contrast=False)