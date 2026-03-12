import ast
import argparse
import torch
import subprocess
import sys
import os
from datetime import datetime


# ---- Shared model loading utilities ----

def get_ckpt(ckpt_id, ckpt_path):
    '''Get most recent model checkpoint from a training run directory.'''
    ckpt_dir = f"{ckpt_path}/{ckpt_id}/checkpoints/"
    ckpt_paths = [f'{ckpt_dir}/{f}' for f in os.listdir(ckpt_dir)]
    ckpt_paths.sort(key=os.path.getmtime)
    return ckpt_paths[-1]


def get_model_and_tokenizer(configs, device):
    '''Load model weights and tokenizer and return both.'''
    model_ckpt = get_ckpt(configs['model_id'], configs['ckpt_path'])

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training', 'MVTM'))
    from tokenizer import Tokenizer
    from mvtm_tokenized import Tokenized_MVTM

    # Load tokenizer (VQGAN)
    tokenizer = Tokenizer(**configs['tokenizer'])
    tokenizer.if_tokenizer = tokenizer.if_tokenizer.to(device)
    if configs['tokenizer']['he_config_path']:
        tokenizer.he_tokenizer = tokenizer.he_tokenizer.to(device)

    # Load model (MVTM)
    model = Tokenized_MVTM(**configs['model']).load_from_checkpoint(model_ckpt, **configs['model'])

    return model.to(device).eval(), tokenizer


def get_dataloader(val_file, batch_size, dataset_size, downscale=False, remove_background=False,
                   remove_he=False, shuffle=False, rescale=True, deconvolve_he=True):
    '''Get data, returns pytorch dataloader.'''
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from data import get_panel_selection_data
    if type(val_file) == str:
        val_file = [val_file]
    return get_panel_selection_data(
        val_file=val_file,
        batch_size=batch_size,
        dataset_size=dataset_size,
        remove_background=remove_background,
        remove_he=remove_he,
        downscale=downscale,
        shuffle=shuffle,
        rescale=rescale,
        deconvolve_he=deconvolve_he
    )


# ---- CLI argument helpers ----

def parse_list(arg):
    try:
        arg = ast.literal_eval(arg)
        if not isinstance(arg, list):
            raise ValueError("The argument must be a list.")
        return arg
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list format")


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true',):
        return True
    elif v.lower() in ('false',):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def format_and_print_args(args, parser):
    print("---------------------------------- Options --------------------------------")
    for action in parser._actions:
        attribute = getattr(args, action.dest, 'None')
        default = action.default
        attribute_str = str(attribute)
        default_str = str(default)
        line = f"{action.dest}: {attribute_str:30}"
        if attribute != default and default is not None:
            line += f"\t[default: {default_str}]"
        print(line)
    print("\n")


def int_or_string(value):
    if value in ['all', 'All']:
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is an invalid value. It must be an integer or 'all'/'All'.")


def print_vertical_grid(dictionary):
    max_length = max(len(marker) for marker in dictionary.values())
    border = '+' + '-' * (max_length + 8) + '+'
    print(border)
    for key, value in dictionary.items():
        print(f"| {key}: {value.ljust(max_length)} |")
    print(border)


def check_gpu_availability(gpu_id):
    print("---------------------------------- Checking GPU Availability --------------------------------")

    if torch.cuda.is_available() and gpu_id is not None and gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"-- Resource allocated to GPU {gpu_id} successfully!")

        gpu_properties = torch.cuda.get_device_properties(gpu_id)
        try:
            smi_output = subprocess.check_output(f"nvidia-smi -i {gpu_id}", shell=True).decode()
            print(smi_output)
        except subprocess.CalledProcessError as e:
            print("Failed to execute nvidia-smi command:", e)
    else:
        device = torch.device('cpu')
        print(f"-- GPU {gpu_id} not available. Using CPU instead.")

    print("\n")
    return device


def create_panel_order_directories(val_dataset, project_path, checkpoint, run_name):
    print("---------------------------------- Creating Directories --------------------------------")
    val_dataset_name = '_'.join(val_dataset)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = current_time + "-" + run_name

    panel_order_dir_name = f"panel_order_model={os.path.basename(os.path.dirname(checkpoint))}_dataset={val_dataset_name}"
    panel_order_dir_path = os.path.join(project_path, panel_order_dir_name, current_time)

    if not os.path.exists(panel_order_dir_path):
        os.makedirs(panel_order_dir_path)
        print("-- Creating panel order directory: ", panel_order_dir_path, "\n")

    print("\n")
    return val_dataset_name, panel_order_dir_path


def create_intensity_directories(val_dataset, project_path, checkpoint, run_name, save_pred):
    print("---------------------------------- Creating Directories --------------------------------")
    val_dataset_name = '_'.join(val_dataset)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = current_time + "-" + run_name

    intensity_feature_dir_name = f"intensity_features_model={os.path.basename(os.path.dirname(checkpoint))}_dataset={val_dataset_name}"
    intensity_feature_dir_path = os.path.join(project_path, intensity_feature_dir_name, current_time)

    pred_img_dir_path = None

    if not os.path.exists(intensity_feature_dir_path):
        os.makedirs(intensity_feature_dir_path)
        print("-- Creating intensity feature directory: ", intensity_feature_dir_path, "\n")

    if save_pred:
        pred_img_dir_name = f"saved_pred_images_model={os.path.basename(os.path.dirname(checkpoint))}_dataset={val_dataset_name}"
        pred_img_dir_path = os.path.join(project_path, pred_img_dir_name, current_time)
        if not os.path.exists(pred_img_dir_path):
            os.makedirs(pred_img_dir_path)
            print("-- Creating predicted images directory: ", pred_img_dir_path)

    print("\n")
    return val_dataset_name, intensity_feature_dir_path, pred_img_dir_path


def create_metadata_file(panel_ch_idx, masked_ch_idx, current_marker_panel, ordered_markers_to_impute, file_path, size):
    combined_ch_idx = panel_ch_idx + masked_ch_idx
    combined_marker_names = current_marker_panel + ordered_markers_to_impute

    metadata_content = f"Channel Index and Name Information for panel={size}\n"
    metadata_content += "----------------------------------\n"
    metadata_content += "Feature intensity table index order: (Panel Marker Index || Imputed Marker Index || Coordinates):\n"
    concatenated_indices = ', '.join(map(str, panel_ch_idx)) + " || " + ', '.join(map(str, masked_ch_idx)) + " || Coordinates"
    metadata_content += concatenated_indices + "\n\n"
    metadata_content += "Feature intensity table channel name order: (Panel Marker Name || Imputed Marker Name || Coordinates):\n"
    concatenated_names = ', '.join(map(str, current_marker_panel)) + " || " + ', '.join(map(str, ordered_markers_to_impute)) + " || Coordinates"
    metadata_content += concatenated_names + "\n\n"
    metadata_content += "Panel Channels:\n"
    for idx, name in zip(panel_ch_idx, current_marker_panel):
        metadata_content += f"Index: {idx}, Name: {name}\n"
    metadata_content += "\nImputed Channels:\n"
    for idx, name in zip(masked_ch_idx, ordered_markers_to_impute):
        metadata_content += f"Index: {idx}, Name: {name}\n"

    with open(file_path, 'w') as file:
        file.write(metadata_content)
