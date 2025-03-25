import ast
import argparse
import torch
import subprocess
import sys
import os
from datetime import datetime

# helper function to parse string to list 
def parse_list(arg):
    try:
        arg = ast.literal_eval(arg)
        if not isinstance(arg, list):
            raise ValueError("The argument must be a list.")
        return arg
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list format")

# helper function to parse string to boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# helper function to print the passed the arguments
def format_and_print_args(args, parser):
    print("---------------------------------- Options --------------------------------")
    for action in parser._actions:
        # Retrieve the attribute name and its value from args
        attribute = getattr(args, action.dest, 'None')
        default = action.default
        attribute_str = str(attribute)
        default_str = str(default)

        # Format and print the argument
        line = f"{action.dest}: {attribute_str:30}"
        if attribute != default and default is not None:
            line += f"\t[default: {default_str}]"
        print(line)
    # print("---------------------------------- End Options --------------------------------\n")
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

def print_warning(warning_message): 
    print("\n" + "*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + warning_message.center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80 + "\n")



def check_gpu_availability(gpu_id):

    print("---------------------------------- Checking GPU Availability --------------------------------")
    
    if torch.cuda.is_available() and gpu_id is not None and gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"-- Resource allocated to GPU {gpu_id} successfully!")

        # Get basic GPU properties using PyTorch
        gpu_properties = torch.cuda.get_device_properties(gpu_id)
        
        # Get more detailed info using a system call to nvidia-smi
        try:
            smi_output = subprocess.check_output(f"nvidia-smi -i {gpu_id}", shell=True).decode()
            print(smi_output)
        except subprocess.CalledProcessError as e:
            print("Failed to execute nvidia-smi command:", e)

    else:
        device = torch.device('cpu')
        print(f"-- Resource allocated to GPU {gpu_id} failed! Using CPU instead\n")

        # Prompting the user for input
        while True:
            user_decision = input("Do you want to proceed with CPU [y/n]? ").lower().strip()
            if user_decision == 'y':
                print("Continuing with CPU...")
                break
            elif user_decision == 'n':
                print("Exiting...")
                sys.exit()  # Terminate the program
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")

    print("\n")
    return device



def check_panel_size(include_heatmap, panel_sizes, panel_order):
    if include_heatmap: 
        if panel_sizes != [i for i in range(1, len(panel_order) + 1)]:
            warning_message = "WARNING: Wrong panel_size input! Please check your input!"
            print_vertical_grid(warning_message)
            print("Heatmap analysis requires panel_sizes to have the same length as panel_order")
            print("Example:")
            print("panel_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]")
            print("panel_order: [0, 4, 12, 10, 1, 11, 16, 8, 3, 5, 9, 6, 2, 13, 7, 14]")
            
            while True:
                user_response = input("Do you want to continue without generating the heatmap [y/n]?: ").strip().lower()
                if user_response == 'y':
                    print("Continuing with analysis...Skipping heatmap generation...")
                    include_heatmap = False
                    break  # Exit the while loop and continue with analysis
                elif user_response == 'n':
                    print("Exiting analysis...")
                    sys.exit()  # Terminate the program
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    
    return include_heatmap


def create_panel_order_directories(val_dataset, project_path, checkpoint, run_name):

    print("---------------------------------- Creating Directories --------------------------------")

    val_dataset_name = '_'.join(val_dataset)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = current_time + "-" + run_name  # append the identifier name after current time
    
    # Creating a new directory for storing the intensity feature tables    
    panel_order_dir_name = f"panel_order_model={os.path.basename(os.path.dirname(checkpoint))}_dataset={val_dataset_name}"
    panel_order_dir_path = os.path.join(project_path, panel_order_dir_name, current_time)

    # Create the intensity feature directory if it does not exist
    if not os.path.exists(panel_order_dir_path):
        os.makedirs(panel_order_dir_path)
        print("-- Creating panel order directory: ", panel_order_dir_path, "\n")
    
    print("\n")
    
    return val_dataset_name, panel_order_dir_path




def create_intensity_directories(val_dataset, project_path, checkpoint, run_name, save_pred):

    print("---------------------------------- Creating Directories --------------------------------")

    val_dataset_name = '_'.join(val_dataset)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = current_time + "-" + run_name  # append the identifier name after current time
    
    # Creating a new directory for storing the intensity feature tables    
    intensity_feature_dir_name = f"intensity_features_model={os.path.basename(os.path.dirname(checkpoint))}_dataset={val_dataset_name}"
    intensity_feature_dir_path = os.path.join(project_path, intensity_feature_dir_name, current_time)
    
    # Initialize pred_img_dir_path as None
    pred_img_dir_path = None

    # Create the intensity feature directory if it does not exist
    if not os.path.exists(intensity_feature_dir_path):
        os.makedirs(intensity_feature_dir_path)
        print("-- Creating intensity feature directory: ", intensity_feature_dir_path, "\n")

    if save_pred:
        # Create a new directory for storing the saved predicted and ground truth images 
        pred_img_dir_name = f"saved_pred_images_model={os.path.basename(os.path.dirname(checkpoint))}_dataset={val_dataset_name}"
        pred_img_dir_path = os.path.join(project_path, pred_img_dir_name, current_time)
        if not os.path.exists(pred_img_dir_path):
            os.makedirs(pred_img_dir_path)
            print("-- Creating predicted images directory: ", pred_img_dir_path)

    print("\n")
    return val_dataset_name, intensity_feature_dir_path, pred_img_dir_path


def create_metrics_directory(val_dataset, project_path, checkpoint_name, intensity_feature_dir, run_name):
    """
    Creates a directory for metrics evaluation.

    Args:
        val_dataset (list): List of validation datasets.
        project_path (str): Path to the project directory.
        checkpoint_name (str): Name of the model checkpoint.
        intensity_feature_dir (str): Directory path of intensity features.
        run_name (str): Name of the run.

    Returns:
        str: Path to the created metrics evaluation directory.
    """

    print("---------------------------------- Creating Directories --------------------------------")
    
    val_dataset_name = '_'.join(val_dataset)
    metrics_dir_name = f"metrics_evaluation_model={checkpoint_name}_dataset={val_dataset_name}"
    current_time = '-'.join(os.path.basename(intensity_feature_dir).split('-')[:2]) # remind user not to modify folder name
    current_time = current_time + "-" + run_name
    metrics_dir_path = os.path.join(project_path, metrics_dir_name, current_time)

    if not os.path.exists(metrics_dir_path):
        os.makedirs(metrics_dir_path)
        print("-- Creating metrics evaluation directory: ", metrics_dir_path)

    return metrics_dir_path


def create_metadata_file(panel_ch_idx, masked_ch_idx, current_marker_panel, ordered_markers_to_impute, file_path, size):
    """
    Creates a metadata file for mean intensity feature table.

    Args:
        panel_ch_idx (list): List of unmasked channel indices.
        masked_ch_idx (list): List of masked channel indices.
        current_marker_panel (list): Names of unmasked channels.
        ordered_markers_to_impute (list): Names of masked channels.
        file_path (str): Path to save the metadata file.
    """
    # Combine indices and names
    combined_ch_idx = panel_ch_idx + masked_ch_idx
    combined_marker_names = current_marker_panel + ordered_markers_to_impute

    # Create metadata content
    metadata_content = f"Channel Index and Name Information for panel={size}\n"
    metadata_content += "----------------------------------\n"

    metadata_content += "Feature intensity table index order: (Panel Marker Index || Inputed Marker Index || Coordinates):\n"
    concatenated_indices = ', '.join(map(str, panel_ch_idx)) + " || " + ', '.join(map(str, masked_ch_idx)) + " || Coordinates"
    metadata_content += concatenated_indices + "\n\n"

    metadata_content += "Feature intensity table channel name order: (Panel Marker Name || Inputed Marker Name || Coordinates):\n"
    concatenated_names = ', '.join(map(str, current_marker_panel)) + " || " + ', '.join(map(str, ordered_markers_to_impute)) + " || Coordinates"
    metadata_content += concatenated_names + "\n\n"

    metadata_content += "Panel Channels:\n"
    for idx, name in zip(panel_ch_idx, current_marker_panel):
        metadata_content += f"Index: {idx}, Name: {name}\n"
    
    metadata_content += "\nImputed Channels:\n"
    for idx, name in zip(masked_ch_idx, ordered_markers_to_impute):
        metadata_content += f"Index: {idx}, Name: {name}\n"

    # Write to file
    with open(file_path, 'w') as file:
        file.write(metadata_content)
