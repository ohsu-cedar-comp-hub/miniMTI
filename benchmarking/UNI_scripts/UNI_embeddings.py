import os
import h5py
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from uni import get_encoder

device = "cuda:4"
local_dir = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/assets/ckpts_custom"
os.makedirs(local_dir, exist_ok=True)

# Load model and transform
model, transform = get_encoder(enc_name='uni2-h', device=device)

h5_path = "/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized-All/orion_crc_dataset_sid=CRC13.h5"
out_path = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/validation_embeddings_and_means/CRC13/UNI_embeddings_and_means.h5"
BATCH_SIZE = 64
MAX_ITEMS = 1014386

def compute_mean_intensities(if_images, mask=None):
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        return (if_images * mask).sum((2, 3)) / mask.sum((2, 3)).clamp(min=1.)
    return if_images.mean((2, 3))


with h5py.File(h5_path, 'r') as f_in, h5py.File(out_path, 'w') as f_out:
    total_samples = min(f_in['images'].shape[0], MAX_ITEMS)
    print('Creating empty outputs')
    # Create output datasets
    dset_emb = f_out.create_dataset("embeddings", shape=(total_samples, 1536), dtype='float32')
    dset_means = f_out.create_dataset("all_if_means", shape=(total_samples, 17), dtype='float32')
    print('Starting inference')
    with torch.inference_mode():
        for start in range(0, total_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_samples)
            # print('Loading Batch')
            batch_imgs = f_in['images'][start:end]
            batch_masks = f_in['masks'][start:end]
            batch_he = batch_imgs[:, :, :, -3:]  # Last 3 channels (HE)
            # print('Loaded Batch')            
            # def preprocess(i):
            #     print('Start preprocess')
            #     he_img = batch_he[i].transpose(2, 0, 1)  # HWC -> CHW
            #     pil_img = Image.fromarray(he_img.transpose(1, 2, 0).astype("uint8"))
            #     print('End preprocess')
            #     return transform(pil_img)
            def preprocess(i):
                try:
                    print(f'Start preprocess {i}')
                    he_img = batch_he[i].transpose(2, 0, 1)  # HWC -> CHW
                    print(he_img.shape)
                    pil_img = Image.fromarray(he_img.transpose(1, 2, 0).astype("uint8"))
                    out = transform(pil_img)
                    print(f'End preprocess {i}')
                    return out
                except Exception as e:
                    print(f"Error in preprocess {i}: {e}")
                    raise
                        
            with ThreadPoolExecutor() as executor:
                batch_tensors = list(executor.map(preprocess, range(batch_he.shape[0])))
            
            batch_tensors = torch.stack(batch_tensors).to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                print('autocast')
                batch_embeddings = model(batch_tensors).cpu().numpy()
            # print('Loading if masks')
            if_images = torch.from_numpy(batch_imgs[:, :, :, :17]).permute(0, 3, 1, 2).float()
            batch_masks_t = torch.from_numpy(batch_masks).float()
            all_if_means_batch = compute_mean_intensities(if_images, batch_masks_t)
            # print('Loaded if masks')            
            dset_emb[start:end] = batch_embeddings
            # print('Loaded emb')    
            dset_means[start:end] = all_if_means_batch.cpu().numpy()             
            print(f"Processed {end}/{total_samples}")
            
            del batch_embeddings, if_images, batch_masks_t, all_if_means_batch, batch_tensors
            torch.cuda.empty_cache()
