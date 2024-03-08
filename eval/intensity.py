import os
import sys
import gc
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rescale
from scipy.stats import gaussian_kde
from einops import rearrange, repeat
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman

def get_mints(unmasked, gt, preds, mask, device):
    #reshape to BxCxHxW
    preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
    gt = rearrange(gt, 'b c (h w) -> b c h w', h=32)

    #expand dapi mask to number of reconstructed channels
    mask = repeat(mask,'b h w -> b c h w', c=preds.shape[1])
    mask = mask.to(device)

    #calculate mean intensities
    mints = (gt * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
    pmints = (preds * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))

    return mints, pmints



def get_intensities_run_panel_order(model, panel, val_loader, max_panel_size, batch_size, device='cpu'):
    
    masked_ch_idx = torch.tensor([i for i in range(max_panel_size) if i not in panel], device=device)
    model.mae.masking_ratio = (max_panel_size - len(panel)) / max_panel_size

    with torch.no_grad():
        #set tensors on cpu to store predicted and ground-truth mean intensities 
        mints= torch.zeros((len(val_loader.dataset), len(masked_ch_idx)), device='cpu')
        pmints = mints.clone()

        #run each batch through the model and copy intensities back to cpu
        for i, (ims, mask, filepaths) in enumerate(tqdm(val_loader)):
            ims = ims.to(device)
            panel_patches, gt, preds,_,_ = model.forward((ims, mask, filepaths), masked_patch_idx=masked_ch_idx)
            mints_, pmints_ = get_mints(ims, gt, preds, mask, device)
            s = i * batch_size
            e = s + batch_size
            mints_ = mints_.to('cpu')
            pmints_ = pmints_.to('cpu')
            mints[s:e, :] = mints_
            pmints[s:e, :] = pmints_

            del ims
            if device != 'cpu':
                gc.collect()
                torch.cuda.empty_cache()
                     
    return mints, pmints


def get_intensities(model, panel, val_loader, max_panel_size, calculate_ssims, include_he, save_pred, pred_img_dir_path, size, device='cpu'):

    # if include_he: max_panel_size += 3
    keep_channels, keep_channels_idx, ch2idx = get_channel_info(include_he=include_he)
    ch2stain = {i:ch for ch,i in ch2idx.items()}

    ordered_markers_to_impute = [ch2stain[idx] for idx in [i for i in range(max_panel_size) if i not in panel]]
    current_marker_panel = [ch2stain[idx] for idx in panel]
    masked_ch_idx_list = [i for i in range(max_panel_size) if i not in panel]
    masked_ch_idx = torch.tensor(masked_ch_idx_list, device=device)
    panel_ch_idx = torch.tensor(panel, device=device)  # This is the panel list, in this above example, this will be tensor([0, 4, 12])
    model.mae.masking_ratio = (max_panel_size - len(panel)) / max_panel_size

    print(f"*************** Processing {size} panel ***************")
    print("-- Current marker panel is:", current_marker_panel)
    print("-- Current panel marker index: ", panel)
    print("-- Markers to impute: ", ordered_markers_to_impute)
    print("-- Markers to impute index: ", [i for i in range(max_panel_size) if i not in panel])
    print("-- Calculating SSIM: ", calculate_ssims)
    print("-- Current MAE masking ratio is: ", model.mae.masking_ratio)
    

    with torch.no_grad():
        # Initialize tensors to store results
        # this will create a zero matrix of shape (dataset_size, masked_ch_idx_size), in the above example, this will be (50000, 14) 
        mints = torch.zeros((len(val_loader.dataset), len(masked_ch_idx)), device='cpu')
        pmints = mints.clone()
        # set up the combined mint and pmint
        # the additional 2 is for the centroid information
        mints_combined = torch.zeros((len(val_loader.dataset), max_panel_size + 2), device='cpu')
        pmints_combined = mints_combined.clone()
        # set up the ssims
        ssims = torch.zeros(len(val_loader.dataset), device='cpu') if calculate_ssims else None

        # creating list to store pred and gt images
        all_preds = []
        all_gt = []

        print(f"-- Processing batches ......")
        
        batch_size = None
        for i, batch in enumerate(tqdm(val_loader)):
            ims, masks, filepaths = batch
            # Process batch and mask 
            # print(f"-- Processing batch {i} ......")
            ims = ims.to(device)
            if batch_size is None: 
                batch_size = len(ims)

            # get centroid information from filepaths
            # centroids will be a list with a dimension of (batch_size, 2)
            # print("-- Getting the centroid information......")
            centroids = []
            for filepath in filepaths:
                parts = filepath.split('-')
                x_part = parts[-2]
                y_part = parts[-1]
                x_centroid = int(x_part.split('=')[1])
                y_centroid = int(y_part.split('=')[1].replace('.tif', ''))
                centroids.append((x_centroid, y_centroid))
            centroids_tensor = torch.tensor(centroids, device=device)
        
            panel_patches, gt, preds, _, _ = model.forward((ims, masks, filepaths), masked_patch_idx=masked_ch_idx)
            concatenated_gt = torch.cat((panel_patches, gt), dim=1)
            concatenated_preds = torch.cat((panel_patches, preds), dim=1)

            # reshape gt and preds from torch.Size([batch_size, 16, 1024]) to torch.Size([batch_size, 16, 32, 32])
            preds_reshape = rearrange(preds, 'b c (h w) -> b c h w', h=32)
            gt_reshape = rearrange(gt, 'b c (h w) -> b c h w', h=32)

            mints_, pmints_ = get_mints(ims, gt, preds, masks, device) # without panel patches 
            mints_concat_, pmints_concat_ = get_mints(ims, concatenated_gt, concatenated_preds, masks, device) # with panel patches

            mints_combined_ = torch.cat((mints_concat_, centroids_tensor), dim=1)
            pmints_combined_ = torch.cat((pmints_concat_, centroids_tensor), dim=1)

            s = i * batch_size
            e = s + batch_size

            mints[s:e, :] = mints_.to('cpu')
            pmints[s:e, :] = pmints_.to('cpu')

            mints_combined[s:e, :] = mints_combined_.to('cpu')
            pmints_combined[s:e, :] = pmints_combined_.to('cpu')

            if save_pred: 
                all_preds.append(preds_reshape.to('cpu'))
                all_gt.append(gt_reshape.to('cpu'))

            # SSIM calculation if requested
            if calculate_ssims:
                # preds and gt now will have a shape of torch.Size([batch_size, 16, 32, 32])
                # preds_reshape = rearrange(preds, 'b c (h w) -> b c h w', h=32)
                preds_reshape[preds_reshape < 0] = 0
                # gt_reshape = rearrange(gt, 'b c (h w) -> b c h w', h=32)
                ssims_ = ssim(gt_reshape, preds_reshape, reduction='none')
                ssims_ = ssims_.to('cpu')
                ssims[s:e] = ssims_
                 
            del batch
            if device != 'cpu':
                gc.collect()
                torch.cuda.empty_cache()

        if save_pred:
            combined_preds = torch.cat(all_preds, dim=0)
            combined_gt = torch.cat(all_gt, dim=0)
    
            print(f"-- Saving predicted images with shape {list(combined_preds.shape)} to {os.path.join(pred_img_dir_path, f'pred_images_size={combined_preds.shape[0]}_{size}panel.pt')}......")
    
            print(f"-- Saving ground truth images with shape {list(combined_gt.shape)} to {os.path.join(pred_img_dir_path, f'gt_images_size={combined_gt.shape[0]}_{size}panel.pt')}......")
    
            torch.save(combined_preds, os.path.join(pred_img_dir_path, f'pred_images_size={combined_preds.shape[0]}_{size}panel.pt'))
            torch.save(combined_gt, os.path.join(pred_img_dir_path, f'gt_images_size={combined_gt.shape[0]}_{size}panel.pt'))
        
    return (mints, pmints, mints_combined, pmints_combined, ssims, panel, masked_ch_idx_list, current_marker_panel, ordered_markers_to_impute)


def plot_intensities(mints, pmints, masked_ch_idx, ch2stain):
    mints, pmints = mints, pmints
    if mints.shape[1] == 1:
        stain_corrs = spearman(pmints.squeeze(), mints.squeeze())
        stain_corrs = stain_corrs.unsqueeze(0)
    else:
        stain_corrs = spearman(pmints, mints)
    #cls iteratively selected panel
    fig, ax = plt.subplots(1, len(masked_ch_idx), figsize=(12 * len(masked_ch_idx), 12), layout='tight')
    for i,a in enumerate(fig.axes):

        a.set_title(f'{ch2stain[masked_ch_idx[i]]}\n(⍴={round(stain_corrs[i].item(),2)})', fontsize=120)
        x,y = mints[:20000,i].cpu(), pmints[:20000,i].cpu()
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        a.scatter(x, y, c=z, s=10)
        a.set_xticks([])
        a.set_yticks([])
        a.plot(np.arange(255), np.arange(255), linestyle='dashed', c='black')
    plt.show()
    return stain_corrs
