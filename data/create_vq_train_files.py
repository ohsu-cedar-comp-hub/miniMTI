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

#savedir = '/mnt/scratch/orion-vq-if-train-images'
#savedir = '/mnt/scratch/lunaphore-vq-if-train-images'
#savedir = '/mnt/scratch/orion-vq-he-train-images'
savedir = '/mnt/scratch/orion-vq-if-he-train-images'
sids = ['CRC01','CRC02','CRC03','CRC04','CRC08','CRC11']
#sid = '1010332'
for sid in sids:
    data = h5py.File(f'/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized-All/orion_crc_dataset_sid={sid}.h5')
    #data = h5py.File(f'/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized-Final/orion_crc_dataset_sid={sid}.h5')
    #data = h5py.File(f'/home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_dataset_norm_sid={sid}.h5')

    for i,im in tqdm(enumerate(data['images'][:100_000])):
        #for c in range(43):
        for c in range(17):
            #imsave(f'{savedir}/{sid}-sample-{i}-ch-{c}.tif', im[:,:,c], check_contrast=False)
            np.save(f'{savedir}/{sid}-sample-{i}-ch-{c}.npy', im[:,:,c])


        he_im = im[:,:,-3:]
        deconvolved_im = deconvolve(he_im)
        for c in range(2):
            np.save(f'{savedir}/{sid}-sample-{i}-he-ch-{c}.npy', deconvolved_im[:,:,c])

        #imsave(f'{savedir}/{sid}-sample-{i}-HE.tif', he_im.astype('uint8'), check_contrast=False)