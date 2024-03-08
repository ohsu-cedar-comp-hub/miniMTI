import os
import sys
import gc
import argparse
import json
from datetime import datetime
import torch
from torchmetrics.functional import spearman_corrcoef as spearman
sys.path.append('../data')
from data import get_panel_selection_data
from eval_mae import IF_MAE
from intensity import get_intensities_run_panel_order
from process_cedar_biolib_immune import get_channel_info
from helper import parse_list, str2bool, format_and_print_args, int_or_string, print_vertical_grid, check_gpu_availability, create_intensity_directories, create_metadata_file, create_panel_order_directories

import warnings
warnings.filterwarnings("ignore")


def get_channel_order(max_panel_size, val_loader, model, ch2stain, batch_size):
    
    device = model.device
    top_panel = [0]
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

            mints, pmints = get_intensities_run_panel_order(model, candidate_panel, val_loader, max_panel_size, batch_size, device=device)
            
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
    return top_panel


def main(run_name, project_path, val_dataset, val_dataset_size, remove_background, ckpt, max_panel_size, gpu_id, batch_size, reverse_channel_order, param_file): 
    
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
        print("-- Loading testing dataset", val_dataset, "......")
        val_loader = get_panel_selection_data(val_dataset, batch_size, val_dataset_size, remove_background) 
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
    if reverse_channel_order:
        top_panel = get_channel_order_reversed(val_loader, model, ch2stain)
    else:
        top_panel = get_channel_order(max_panel_size, val_loader, model, ch2stain, batch_size)

    val_dataset_name, panel_order_dir_path = create_panel_order_directories(val_dataset, project_path, ckpt, run_name)

    output_filename = f'{val_dataset_name}_panel_order.txt'
    output_path = os.path.join(panel_order_dir_path, output_filename)
    with open(output_path, 'w') as f:
        f.write(str(top_panel))
    print(f"Panel order output saved to {output_path}")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with panel reduction model')
    parser.add_argument('--run_name', type=str, required=True, help='give an identifier to your current inference run')
    parser.add_argument('--project_path', type=str, default='/home/groups/ChangLab/wangmar/shift-panel-reduction/shift-panel-reduction-main/eval', help='path to the panel reduction project')
    parser.add_argument("--val_dataset", type=str, nargs='*', required=True, help="Which CRC dataset to use for testing/validation, e.g. CRC05")
    parser.add_argument("--val_dataset_size", type=int_or_string, default="all", help="The size of the validation dataset to be used for inferencing")
    parser.add_argument("--remove_background", action='store_true', help="Whether to remove background information, e.g. True")
    parser.add_argument("--ckpt", type=str, required=True, help="file path to the checkpoint")
    parser.add_argument("--max_panel_size", type=int, default=17, help="Maximum panel size for Orion-CRC dataset, 17 for no HE and 20 for HE")
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use (if available)')
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for processing")
    parser.add_argument('--reverse_channel_order', action='store_true',
                        help='If True, use get_channel_order_reversed(), otherwise use get_channel_order()')   
    parser.add_argument("--param_file", type=str, default=None, help="json file containing model hyperparameters")
    args = parser.parse_args()
    
    format_and_print_args(args, parser) # Print the formatted arguments
    main(args.run_name, args.project_path, args.val_dataset, args.val_dataset_size, args.remove_background, args.ckpt, args.max_panel_size, args.gpu_id, args.batch_size, args.reverse_channel_order, args.param_file)

    print("########## Run Panel Order Complete ##########")
