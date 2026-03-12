import os
import sys
import gc
import argparse
import json
import torch
import numpy as np
from torchmetrics.functional import spearman_corrcoef as spearman

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data import get_panel_selection_data
from intensity_tokenized import get_intensities
from crc_orion_channel_info import get_channel_info
from helper import (
    get_model_and_tokenizer, parse_list, str2bool,
    format_and_print_args, int_or_string, print_vertical_grid,
    check_gpu_availability, create_intensity_directories,
    create_metadata_file, create_panel_order_directories
)

import warnings
warnings.filterwarnings("ignore")


def get_channel_order(max_panel_size, val_loader, model, tokenizer, ch2stain, batch_size):
    """Find most informative marker first (marker that best predicts n - 1 markers)."""
    device = model.device
    top_panel = [17]
    candidate_scores = []
    for num_masked in reversed(range(1, max_panel_size - len(top_panel))):
        top_corr = -999
        top_ch = None
        for ch_candidate in range(max_panel_size):
            if ch_candidate in top_panel:
                continue
            candidate_panel = top_panel.copy()
            candidate_panel.append(ch_candidate)
            print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')

            mints, pmints, _ = get_intensities(
                model=model,
                tokenizer=tokenizer,
                panel=candidate_panel,
                val_loader=val_loader,
                ch2stain=ch2stain,
                calculate_ssims=False,
                device=device,
                return_meta=False,
                T=1
            )

            if mints.shape[1] == 1:
                mints = mints.squeeze()
                pmints = pmints.squeeze()
            corr = torch.mean(spearman(pmints, mints))
            print(f'{corr=}')

            candidate_scores.append((candidate_panel, corr))

            if corr > top_corr:
                print('found new top panel')
                top_corr = corr
                top_ch = ch_candidate

            gc.collect()
            torch.cuda.empty_cache()

        top_panel.append(top_ch)
    return top_panel, candidate_scores


def get_channel_order_reversed(max_panel_size, val_loader, model, tokenizer, ch2stain, batch_size):
    """Find easiest marker first (easiest marker to predict from n - 1 markers)."""
    device = model.device
    top_panel = []
    for num_masked in range(1, max_panel_size - 1):
        top_corr = -999
        top_ch = None
        for ch_candidate in range(max_panel_size):
            if ch_candidate in top_panel or ch_candidate == 17:
                continue
            candidate_panel = [i for i in range(max_panel_size) if (i not in top_panel) and (i != ch_candidate)]

            mints, pmints, _ = get_intensities(
                model=model,
                tokenizer=tokenizer,
                panel=candidate_panel,
                val_loader=val_loader,
                ch2stain=ch2stain,
                calculate_ssims=False,
                device=device,
                return_meta=False,
                T=1
            )
            if mints.shape[1] == 1:
                mints = mints.squeeze()
                pmints = pmints.squeeze()
            corr = torch.mean(spearman(pmints, mints))

            if corr > top_corr:
                top_corr = corr
                top_ch = ch_candidate

            gc.collect()
            torch.cuda.empty_cache()

        top_panel.append(top_ch)
    top_panel.extend([i for i in range(len(ch2stain)) if i not in top_panel and i != 17])
    top_panel.append(17)
    return list(reversed(top_panel))[:-1], None


def get_channel_order_forward_reverse(max_panel_size, val_loader, model, tokenizer, ch2stain, batch_size):
    """Combined forward and reverse panel ordering."""
    device = model.device
    top_panel = np.ones(max_panel_size) * -999
    top_panel[0] = 0
    forward_i = 1
    backward_i = max_panel_size - 1
    forward = True
    while forward_i < backward_i:
        top_corr = -999
        top_ch = None
        for ch_candidate in range(max_panel_size):
            already_used = [i for i in top_panel if i != -999]
            if ch_candidate in already_used:
                continue
            if forward:
                candidate_panel = top_panel[:forward_i].astype('int').tolist() + [ch_candidate]
            else:
                candidate_panel = top_panel[:forward_i].astype('int').tolist() + [i for i in range(max_panel_size) if (i not in top_panel) and (i != ch_candidate)]
            print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')

            mints, pmints, _ = get_intensities(
                model=model,
                tokenizer=tokenizer,
                panel=candidate_panel,
                val_loader=val_loader,
                ch2stain=ch2stain,
                calculate_ssims=False,
                device=device,
                return_meta=False,
                T=1
            )
            if mints.shape[1] == 1:
                mints = mints.squeeze()
                pmints = pmints.squeeze()
            corr = torch.mean(spearman(pmints, mints))
            print(f'{corr=}')

            if corr > top_corr:
                print('found new top panel')
                top_corr = corr
                top_ch = ch_candidate

            gc.collect()
            torch.cuda.empty_cache()

        if forward:
            top_panel[forward_i] = top_ch
            forward_i += 1
        else:
            top_panel[backward_i] = top_ch
            backward_i -= 1
        print(f' {forward=}, {top_ch=}, {top_corr=}, {top_panel=}')
        forward = not forward

    last_marker_idx = np.where(top_panel == -999)
    top_panel[last_marker_idx] = [i for i in range(max_panel_size) if i not in top_panel][0]
    return top_panel[:-1].astype('int').tolist(), None


def main(dataset, dataset_size, remove_background, max_panel_size, gpu_id, batch_size, reverse, forward_reverse, param_file, downscale=False, remove_he=False, deconvolve_he=False):

    device = check_gpu_availability(gpu_id)

    print("---------------------------------- Collecting Channel Info --------------------------------")
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()

    ch2stain = {i: ch for ch, i in ch2idx.items()}
    print_vertical_grid(ch2stain)
    print("\n")

    try:
        print("---------------------------------- Loading Validation Dataset --------------------------------")
        print("-- Loading testing dataset", dataset, "......")
        val_loader = get_panel_selection_data(dataset, batch_size, dataset_size, remove_background, rescale=True, remove_he=False, deconvolve_he=deconvolve_he, downscale=downscale)
        print("-- Testing dataset loaded successfully! ")
    except Exception as e:
        print("An error occurred while loading the testing dataset:", e)
        sys.exit(1)
    print("\n")

    try:
        print("---------------------------------- Loading Model --------------------------------")
        print("-- Loading model ......")

        with open(param_file, "r") as f:
            configs = json.load(f)
        model, tokenizer = get_model_and_tokenizer(configs, device)

        print("-- Model loaded successfully! ")
    except Exception as e:
        print("-- An error occurred while loading the model:", e)
        sys.exit(1)
    print("\n")

    print("---------------------------------- Getting Panel Order --------------------------------")
    if reverse:
        top_panel, candidate_scores = get_channel_order_reversed(max_panel_size, val_loader, model, tokenizer, ch2stain, batch_size)
    elif forward_reverse:
        top_panel, candidate_scores = get_channel_order_forward_reverse(max_panel_size, val_loader, model, tokenizer, ch2stain, batch_size)
    else:
        top_panel, candidate_scores = get_channel_order(max_panel_size, val_loader, model, tokenizer, ch2stain, batch_size)

    if reverse:
        method_suffix = 'ips-reversed'
    elif forward_reverse:
        method_suffix = 'ips-forward-reverse'
    else:
        method_suffix = 'ips'

    output_filename = f"{dataset[0].split('/')[-1]}_{configs['model_id']}_panel_order_{method_suffix}.txt"
    output_path = os.path.join('orderings', output_filename)
    if not os.path.exists('orderings'):
        os.mkdir('orderings')
    with open(output_path, 'w') as f:
        f.write(str(top_panel))
    print(f"Panel order output saved to {output_path}")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run iterative panel selection with tokenized MVTM')
    parser.add_argument("--val-dataset", type=str, nargs='*', required=True, help="Which dataset to use for panel selection")
    parser.add_argument("--val-dataset-size", type=int_or_string, default="all", help="The number of samples to be used for inferencing")
    parser.add_argument("--remove-background", action='store_true', default=False, help="Whether to remove background information")
    parser.add_argument("--max-panel-size", type=int, default=17, help="Maximum Number of Markers in Panel")
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID to use (if available)')
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument('--reversed', action='store_true',
                        help='If True, use get_channel_order_reversed()')
    parser.add_argument('--forward-reverse', action='store_true',
                        help='If True, use combo of forward and reverse ordering')
    parser.add_argument("--param-file", type=str, required=True, help="json file containing model hyperparameters")
    parser.add_argument("--downscale", action='store_true', help="downscale to 32x32 if input is 64x64")
    parser.add_argument("--remove-he", action='store_true', help="remove last three channels from input")
    parser.add_argument("--deconvolve-he", action='store_true', help="deconvolve H&E channels")

    args = parser.parse_args()
    format_and_print_args(args, parser)

    main(dataset=args.val_dataset, dataset_size=args.val_dataset_size, remove_background=args.remove_background, max_panel_size=args.max_panel_size, gpu_id=args.gpu_id, batch_size=args.batch_size, reverse=args.reversed, forward_reverse=args.forward_reverse, param_file=args.param_file, downscale=args.downscale, remove_he=args.remove_he, deconvolve_he=args.deconvolve_he)

    print("########## Run Panel Order Complete ##########")
