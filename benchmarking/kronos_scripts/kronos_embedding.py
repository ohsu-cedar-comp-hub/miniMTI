import sys
import os
import h5py
import torch
from kronos import create_model_from_pretrained

sys.path.append('/home/groups/ChangLab/govindsa/Panel_Reduction_Project/cycif-panel-reduction/')

model, precision, embedding_dim = create_model_from_pretrained(
    checkpoint_path="hf_hub:MahmoodLab/kronos",
    cache_dir="./model_assets",
)
device = 'cuda:5'
model = model.to(device)

print("Model precision: ", precision)
print("Model embedding dimension: ", embedding_dim)


# File paths & batch size

h5_path = "/home/groups/ChangLab/dataset/ORION-CRC-Unnormalized-All/orion_crc_dataset_sid=CRC04.h5"
out_path = "/home/groups/ChangLab/govindsa/KRONOS/code/results/validation_embeddings_and_means/CRC04/CRC04_kronos_embeddings_and_means.h5"
BATCH_SIZE = 512  # smaller batch to reduce memory
MAX_ITEMS = 1416065  # or smaller if testing

def compute_mean_intensities(if_images, mask=None):
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        return (if_images * mask).sum((2, 3)) / mask.sum((2, 3)).clamp(min=1.)
    return if_images.mean((2, 3))


with h5py.File(h5_path, 'r') as f_in, h5py.File(out_path, 'w') as f_out:
    total_samples = min(f_in['images'].shape[0], MAX_ITEMS)
    num_channels = f_in['images'].shape[-1] - 3  # IF channels (exclude last 3 HE)
    num_if_channels = 17

    # Create output datasets
    dset_emb = f_out.create_dataset(
        "embeddings",
        shape=(total_samples, num_channels, embedding_dim),
        dtype='float32'
    )
    dset_means = f_out.create_dataset(
        "all_if_means",
        shape=(total_samples, num_if_channels),
        dtype='float32'
    )

    with torch.inference_mode():
        for start in range(0, total_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_samples)

            # Load only the current batch from disk
            batch_imgs = f_in['images'][start:end]  # [B, H, W, C]
            batch_masks = f_in['masks'][start:end]  # [B, H, W]
            print(f"Mask min/max: {batch_masks.min()} / {batch_masks.max()}, shape: {batch_masks.shape}")
            # Prepare IF channels
            batch_if = torch.from_numpy(batch_imgs[:, :, :, :-3]).permute(0, 3, 1, 2)
            batch_if = batch_if.to(device, dtype=torch.float32) / 255.0


            # Feature extraction
            _, marker_embeddings, _ = model(batch_if)

            # Compute means
            if_images = torch.from_numpy(batch_imgs[:, :, :, :17]).permute(0, 3, 1, 2).float().to(device)          
            batch_masks_t = torch.from_numpy(batch_masks).float().to(if_images.device)
            all_if_means_batch = compute_mean_intensities(if_images, batch_masks_t)

            # Write directly to disk
            dset_emb[start:end] = marker_embeddings.cpu().numpy()
            dset_means[start:end] = all_if_means_batch.cpu().numpy()

            print(f"Processed {end}/{total_samples}")

            # Free memory
            del batch_if, batch_masks_t, if_images, marker_embeddings, all_if_means_batch
            torch.cuda.empty_cache()
