import os
import sys
import gc
import argparse
import json
import pickle
from datetime import datetime
import torch
import numpy as np
from torchmetrics.functional import spearman_corrcoef as spearman
sys.path.append('../data')
from data import get_panel_selection_data
from eval_mae import IF_MAE
from intensity import get_intensities
#from process_cedar_biolib_immune import get_channel_info
from process_aced_immune import get_channel_info
from helper import parse_list, str2bool, format_and_print_args, int_or_string, print_vertical_grid, check_gpu_availability, create_intensity_directories, create_metadata_file, create_panel_order_directories

import warnings
warnings.filterwarnings("ignore")


#find most informative marker first (marker that best predicts n - 1 markers)
def get_channel_order(max_panel_size, val_loader, model, ch2stain, batch_size):
    
    device = model.device
    top_panel = [0]
    candidate_scores = []
    for num_masked in reversed(range(1, max_panel_size - len(top_panel))):
        masking_ratio = num_masked/max_panel_size
        model.mae.masking_ratio = masking_ratio
        top_corr = -999
        top_ch = None
        for ch_candidate in range(max_panel_size):
            if ch_candidate in top_panel: continue
            candidate_panel = top_panel.copy()
            candidate_panel.append(ch_candidate)
            print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')

            mints, pmints, _ = get_intensities(model, candidate_panel, val_loader, ch2stain, device=device)
            
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


#find easiest marker first (easiest marker to predict from n - 1 markers)
def get_channel_order_reversed(max_panel_size, val_loader, model, ch2stain, batch_size):
    device = model.device
    top_panel = []
    for num_masked in range(1,max_panel_size):
        masking_ratio = num_masked/max_panel_size
        model.mae.masking_ratio = masking_ratio

        top_corr = -999
        top_ch = None
        for ch_candidate in range(max_panel_size):
            if ch_candidate in top_panel: continue
            candidate_panel = [i for i in range(max_panel_size) if (i not in top_panel) and (i != ch_candidate)]
            print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')
            print(f'target marker: {ch2stain[ch_candidate]}')

            mints, pmints, _ = get_intensities(model, candidate_panel, val_loader, ch2stain, device=device)
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

        top_panel.append(top_ch)
    top_panel.extend([i for i in range(len(ch2stain)) if i not in top_panel])
    return list(reversed(top_panel))[:-1], None

def get_channel_order_worst_ch(max_panel_size, val_loader, model, ch2stain, batch_size):
    
    device = model.device
    top_panel = [0]
    candidate_scores = []
    for num_masked in reversed(range(1, max_panel_size - len(top_panel))):
        masking_ratio = num_masked/max_panel_size
        model.mae.masking_ratio = masking_ratio
        candidate_panel = top_panel.copy()
        mints, pmints, _ = get_intensities(model, candidate_panel, val_loader, ch2stain, device=device)
        if mints.shape[1] == 1:
            mints = mints.squeeze()
            pmints = pmints.squeeze()
        corr = spearman(pmints, mints)
        worst_pred_ch = corr.argmin().item()
        print(corr.mean())
        unmasked_chs = [i for i in range(max_panel_size) if i not in top_panel]
        top_panel.append(unmasked_chs[worst_pred_ch])
        

        gc.collect()
        torch.cuda.empty_cache()
        
    return top_panel.astype('int').tolist(), None

def get_channel_order_forward_reverse(max_panel_size, val_loader, model, ch2stain, batch_size):
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
            if ch_candidate in already_used: continue
            if forward:
                candidate_panel = top_panel[:forward_i].astype('int').tolist() + [ch_candidate]
            else:
                candidate_panel = top_panel[:forward_i].astype('int').tolist() + [i for i in range(max_panel_size) if (i not in top_panel) and (i != ch_candidate)]
            print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')
            
            num_masked = max_panel_size - len(candidate_panel)
            masking_ratio = num_masked/max_panel_size
            model.mae.masking_ratio = masking_ratio
            mints, pmints, _ = get_intensities(model, candidate_panel, val_loader, ch2stain, device=device)
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
        
    last_marker_idx = np.where(top_panel==-999)
    top_panel[last_marker_idx] = [i for i in range(max_panel_size) if i not in top_panel][0]
    return top_panel[:-1].astype('int').tolist(), None
    

def main(dataset, dataset_size, remove_background, ckpt, max_panel_size, gpu_id, batch_size, reverse, forward_reverse, param_file): 
    
    device = check_gpu_availability(gpu_id)

    print("---------------------------------- Collecting Channel Info --------------------------------")
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()
    
    ch2stain = {i:ch for ch,i in ch2idx.items()}
    print_vertical_grid(ch2stain) # print the marker order 
    print("\n")

    try: 
        print("---------------------------------- Loading Validation Dataset --------------------------------")
        # We also need cell location information, that is embedded in the image file, we need to get that information
        # and append it to the feature intensity table. 
        print("-- Loading testing dataset", dataset, "......")
        val_loader = get_panel_selection_data(dataset, batch_size, dataset_size, remove_background) 
        print("-- Testing dataset loaded successfully! ")
    except Exception as e:
        print("An error occurred while loading the testing dataset:", e)
        sys.exit(1)  # Exit the program with a non-zero exit code to indicate an error
    print("\n")


    try: 
        print("---------------------------------- Loading Model --------------------------------")
        ckpt += '/checkpoints/'
        model_name = os.listdir(ckpt)[0]
        checkpoint_id = os.path.basename(os.path.dirname(ckpt))
        print("-- Loading model", checkpoint_id, "......")
        if param_file is not None:
            with open(param_file) as f:
                params = json.load(f)
        else:
            params = {'channels':max_panel_size}
            
        model = IF_MAE(**params).load_from_checkpoint(os.path.join(ckpt, model_name), **params).to(device).eval()
        print("-- Model loaded successfully! ")
    except Exception as e:
        print("-- An error occurred while loading the model:", e)
        sys.exit(1)  # Exit the program with a non-zero exit code to indicate an error
    print("\n")

    print("---------------------------------- Getting Panel Order --------------------------------")
    # Decide which function to call based on reverse_channel_order
    if reverse:
        top_panel, candidate_scores = get_channel_order_reversed(max_panel_size, val_loader, model, ch2stain, batch_size)
    elif forward_reverse:
        top_panel, candidate_scores = get_channel_order_forward_reverse(max_panel_size, val_loader, model, ch2stain, batch_size)
    else:
        top_panel, candidate_scores = get_channel_order(max_panel_size, val_loader, model, ch2stain, batch_size)

    #with open('candidate_panel_scores.pickle', 'wb') as f:
    #    pickle.dump(candidate_scores, f)
        
    if reverse:
        method_suffix = 'ips-reversed'
    elif forward_reverse:
        method_suffix = 'ips-forward-reverse'
    else:
        method_suffix = 'ips'
        
    output_filename = f"{dataset[0].split('/')[-1]}_{ckpt.split('/')[-3]}_panel_order_{method_suffix}.txt"
    output_path = os.path.join('orderings', output_filename)
    with open(output_path, 'w') as f:
        f.write(str(top_panel))
    print(f"Panel order output saved to {output_path}")
    print("\n")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with panel reduction model')
    parser.add_argument("--val_dataset", type=str, nargs='*', required=True, help="Which dataset to use for panel selection")
    parser.add_argument("--val_dataset_size", type=int_or_string, default="all", help="The number of samples to be used for inferencing")
    parser.add_argument("--remove_background", action='store_true', default=True, help="Whether to remove background information")
    parser.add_argument("--ckpt", type=str, required=True, help="file path to the checkpoint")
    parser.add_argument("--max_panel_size", type=int, default=17, help="Maximum Number of Markers in Panel")
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use (if available)')
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for processing")
    parser.add_argument('--reversed', action='store_true',
                        help='If True, use get_channel_order_reversed(), otherwise use get_channel_order()')   
    parser.add_argument('--forward-reverse', action='store_true',
                        help='If True, use combo of get_channel_order and get_channel_order_reversed()')   
    parser.add_argument("--param_file", type=str, default=None, help="json file containing model hyperparameters")
    
    args = parser.parse_args()
    format_and_print_args(args, parser) # Print the formatted arguments
    
    main(dataset=args.val_dataset, dataset_size=args.val_dataset_size, remove_background=args.remove_background, ckpt=args.ckpt, max_panel_size=args.max_panel_size, gpu_id=args.gpu_id, batch_size=args.batch_size, reverse=args.reversed, forward_reverse=args.forward_reverse, param_file=args.param_file)

    print("########## Run Panel Order Complete ##########")
