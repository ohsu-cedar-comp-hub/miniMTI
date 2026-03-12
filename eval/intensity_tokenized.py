import os
import sys
import gc
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from torchmetrics.functional import structural_similarity_index_measure as ssim, spearman_corrcoef as spearman

def get_mints(gt, preds, mask, device):
    '''Calculates mean intensities (mints) and predicted mean intensities (pmints).'''
    mask = repeat(mask, 'b h w -> b c h w', c=preds.shape[1])
    mask = mask.to(device)

    mints = (gt * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
    pmints = (preds * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))

    return mints, pmints


def get_spearman(mints, pmints):
    if mints.shape[1] == 1:
        corrs_per_marker = spearman(pmints.squeeze(), mints.squeeze())
        corrs_per_marker = corrs_per_marker.unsqueeze(0)
    else:
        corrs_per_marker = spearman(pmints, mints)
    return corrs_per_marker


def get_intensities(model, tokenizer, panel, val_loader, ch2stain, calculate_ssims=True, device='cpu', return_meta=False, distributed_state=None, T=1, temp=0, return_logits=False, output_panel=None, return_token_ids=False):
    '''Runs inference with model in batches on GPU and copies mean intensities and predicted mean intensities back to CPU.

    Args:
        panel: Input channel indices (kept visible)
        output_panel: Optional output channel indices (to predict). If None, predicts all non-input channels.
        return_token_ids: If True, returns intermediate token IDs before detokenization
    '''
    max_panel_size = len(ch2stain.keys())
    current_marker_panel = [ch2stain[idx] for idx in panel]

    # Determine which channels to predict
    if output_panel is None:
        masked_ch_idx_list = [i for i in range(max_panel_size) if i not in panel]
        use_new_interface = False
    else:
        masked_ch_idx_list = list(output_panel)
        use_new_interface = True

    ordered_markers_to_impute = [ch2stain[idx] for idx in masked_ch_idx_list]
    masked_ch_idx = torch.tensor(masked_ch_idx_list, device=device)
    panel_ch_idx = torch.tensor(panel, device=device)

    print(f"Processing {len(panel)} input marker(s) -> {len(masked_ch_idx_list)} output marker(s)")
    print(f"  Input markers: {current_marker_panel}")
    print(f"  Output markers: {ordered_markers_to_impute}")

    # Initiate empty arrays on CPU
    mints = torch.zeros((len(val_loader.dataset), len(masked_ch_idx)), device='cpu')
    pmints = mints.clone()
    ssims_tensor = torch.zeros(len(val_loader.dataset), device='cpu') if calculate_ssims else None
    meta = []

    # Initialize logits tensor
    if use_new_interface:
        num_processed_channels = len(panel) + len(masked_ch_idx_list)
    else:
        num_processed_channels = model.num_markers

    logits = torch.zeros((len(val_loader.dataset), num_processed_channels * model.max_positions), device='cpu')
    all_preds = torch.zeros((len(val_loader.dataset), model.num_markers + 2, 32, 32), dtype=torch.uint8, device='cpu')

    if return_token_ids:
        all_token_ids = torch.zeros((len(val_loader.dataset), num_processed_channels * model.max_positions), dtype=torch.long, device='cpu')

    batch_size = None
    for i, batch in enumerate(tqdm(val_loader)):
        ims, masks, filepaths = batch
        gt = ims.to(device)

        if batch_size is None:
            batch_size = len(gt)

        with torch.no_grad():
            token_ids = tokenizer.tokenize(gt)

            if use_new_interface:
                out, predicted_token_ids, labels, logits_ = model.predict(
                    token_ids,
                    input_ch_idx=panel_ch_idx,
                    output_ch_idx=masked_ch_idx,
                    T=T,
                    temp=temp,
                    return_logits=return_logits
                )

                full_token_ids = token_ids.clone()
                num_output_channels = len(masked_ch_idx_list)
                output_start_in_pred = len(panel) * model.max_positions
                output_tokens = predicted_token_ids[:, output_start_in_pred:]

                for idx, ch_idx in enumerate(masked_ch_idx_list):
                    start_in_output = idx * model.max_positions
                    end_in_output = (idx + 1) * model.max_positions
                    start_in_full = ch_idx * model.max_positions
                    end_in_full = (ch_idx + 1) * model.max_positions
                    full_token_ids[:, start_in_full:end_in_full] = output_tokens[:, start_in_output:end_in_output]

                preds = tokenizer.detokenize(full_token_ids).reshape(gt.shape)
            else:
                out, predicted_token_ids, labels, logits_ = model.predict(
                    token_ids,
                    masked_ch_idx=masked_ch_idx,
                    T=T,
                    temp=temp,
                    return_logits=return_logits
                )
                preds = tokenizer.detokenize(predicted_token_ids).reshape(gt.shape)

        # Apply normalization
        preds = (preds + 1) * 127.5
        gt = (gt + 1) * 127.5

        # Get mean intensities
        mints_, pmints_ = get_mints(gt, preds, masks, device)
        mints_ = mints_[:, masked_ch_idx]
        pmints_ = pmints_[:, masked_ch_idx]

        # Copy back to CPU
        actual_batch_size = mints_.shape[0]
        s = i * batch_size
        e = s + actual_batch_size
        mints[s:e, :] = mints_.to('cpu')
        pmints[s:e, :] = pmints_.to('cpu')
        if logits_ is not None:
            logits[s:e, :] = torch.softmax(logits_, dim=-1).amax(dim=-1).to('cpu')
        if return_token_ids:
            all_token_ids[s:e, :] = predicted_token_ids.to('cpu')
        meta.extend(filepaths)
        all_preds[s:e, :, :, :] = preds.to(dtype=torch.uint8, device='cpu')

        if calculate_ssims:
            ssims_ = ssim(gt, preds, reduction='none')
            ssims_ = ssims_.to('cpu')
            ssims_tensor[s:e] = ssims_

        del batch
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

    if return_meta:
        if return_token_ids:
            return mints, pmints, ssims_tensor, meta, logits, all_token_ids
        return mints, pmints, ssims_tensor, meta, logits
    if return_token_ids:
        return mints, pmints, ssims_tensor, logits, all_token_ids
    return mints, pmints, ssims_tensor, logits
