import os
import sys
import gc
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from torchmetrics.functional import structural_similarity_index_measure as ssim, spearman_corrcoef as spearman
#from accelerate.utils import gather
#from accelerate.logging import get_logger
#logger = get_logger(__name__, log_level="DEBUG")

def get_mints(gt, preds, mask, device, model_type='mvtm'):
    '''calculates mean intensities (mints) and predicted mean intensities (pmints'''
    #reshape to BxCxHxW
    if model_type != 'mvtm':
        preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
        gt = rearrange(gt, 'b c (h w) -> b c h w', h=32)

    #expand dapi mask to number of reconstructed channels
    mask = repeat(mask,'b h w -> b c h w', c=preds.shape[1])
    mask = mask.to(device)

    #calculate mean intensities
    mints = (gt * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
    pmints = (preds * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))

    return mints, pmints


def get_intensities(model, panel, val_loader, ch2stain, calculate_ssims=False, device='cpu', return_meta=False, model_type='mvtm', distributed_state=None):
    '''runs inference with model in batches on GPU and copies mean intensities and predicted mean intensities (and SSIM if specified) back to CPU'''
    max_panel_size = len(ch2stain.keys())
    ordered_markers_to_impute = [ch2stain[idx] for idx in [i for i in range(max_panel_size) if i not in panel]]
    current_marker_panel = [ch2stain[idx] for idx in panel]
    masked_ch_idx_list = [i for i in range(max_panel_size) if i not in panel]
    masked_ch_idx = torch.tensor(masked_ch_idx_list, device=device)
    panel_ch_idx = torch.tensor(panel, device=device)  # This is the panel list, in this above example, this will be tensor([0, 4, 12])
    if model_type != "mvtm":
        model.mae.masking_ratio = (max_panel_size - len(panel)) / max_panel_size

    print(f"*************** Processing {len(panel)} panel ***************")
    print("-- Current marker panel is:", current_marker_panel)
    print("-- Current panel marker index: ", panel)
    print("-- Markers to impute: ", ordered_markers_to_impute)
    print("-- Markers to impute index: ", [i for i in range(max_panel_size) if i not in panel])
    print("-- Calculating SSIM: ", calculate_ssims)
    if model_type != 'mvtm':
        print("-- Current MAE masking ratio is: ", model.mae.masking_ratio)
    
    #initiate empty arrays on CPU
    mints = torch.zeros((len(val_loader.dataset), len(masked_ch_idx)), device='cpu')
    pmints = mints.clone()
    ssims = torch.zeros(len(val_loader.dataset), device='cpu') if calculate_ssims else None
    meta = []

    print(f"-- Processing batches ......")

    batch_size = None
    for i, batch in enumerate(tqdm(val_loader)):
        ims, masks, filepaths = batch
        gt = ims.to(device)
        
        #get batch size from first batch
        if batch_size is None: 
            batch_size = len(gt)
            
        #run inference on GPU
        with torch.no_grad():
            if model_type == 'mvtm':
                '''
                all_preds = []
                with distributed_state.split_between_processes(gt, apply_padding=True) as in_:
                    out, token_ids, labels = model.forward(in_, masked_ch_idx=masked_ch_idx)
                    preds = model.decode(token_ids, labels, out['logits'], len(in_))
                    all_preds.extend(preds)
                
                all_preds = gather(all_preds)
                preds = torch.stack(all_preds).reshape(gt.shape)
                '''
                T = 1
                temp = 0
                out,token_ids,labels = model.predict(gt, masked_ch_idx=masked_ch_idx, T=T, temp=temp)
                preds = model.detokenize(token_ids, batch_size=gt.shape[0]).reshape(gt.shape)
                
            else:
                panel_patches, gt, preds, _, _ = model.forward((gt, masks, filepaths), masked_patch_idx=masked_ch_idx)

        # reshape gt and preds from torch.Size([batch_size, 16, 1024]) to torch.Size([batch_size, 16, 32, 32])
        if model_type != 'mvtm':
            preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
            gt = rearrange(gt, 'b c (h w) -> b c h w', h=32)

            
        if model_type == 'mvtm':
            preds = (preds + 1) * 127.5
            gt = (gt + 1) * 127.5
        #get mean intensities
        mints_, pmints_ = get_mints(gt, preds, masks, device) # without panel patches 
        
        if model_type == 'mvtm':
            mints_ = mints_[:, masked_ch_idx]
            pmints_ = pmints_[:, masked_ch_idx]

        #copy back to CPU
        s = i * batch_size
        e = s + batch_size
        mints[s:e, :] = mints_.to('cpu')
        pmints[s:e, :] = pmints_.to('cpu')
        meta.extend(filepaths)

        # SSIM calculation if requested
        if calculate_ssims:
            ssims_ = ssim(gt, preds, reduction='none')
            ssims_ = ssims_.to('cpu')
            ssims[s:e] = ssims_

        #memory management
        del batch
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

    if return_meta:
        return mints, pmints, ssims, meta
    return mints, pmints, ssims


def get_spearman(mints, pmints):
    if mints.shape[1] == 1:
        corrs_per_marker = spearman(pmints.squeeze(), mints.squeeze())
        corrs_per_marker = corrs_per_marker.unsqueeze(0)
    else:
        corrs_per_marker = spearman(pmints, mints)
    return corrs_per_marker
    