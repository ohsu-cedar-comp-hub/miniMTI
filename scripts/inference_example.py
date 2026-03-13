"""
Example inference script for CycIF Panel Reduction.

Demonstrates the full pipeline:
1. Download pre-trained model from HuggingFace (or use local paths)
2. Load example HDF5 data
3. Run inference with a specified input panel
4. Output CSV with real vs predicted mean intensities
5. Print Spearman correlations

Usage:
    # Using HuggingFace-hosted model (auto-downloads):
    python scripts/inference_example.py --val-file /path/to/sample.h5

    # Using local model paths:
    python scripts/inference_example.py \
        --val-file /path/to/sample.h5 \
        --local-mvtm-ckpt /path/to/mvtm_model.ckpt \
        --local-if-config /path/to/if_config.yaml \
        --local-if-ckpt /path/to/if_model.ckpt \
        --local-he-config /path/to/he_config.yaml \
        --local-he-ckpt /path/to/he_model.ckpt
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torchmetrics.functional import spearman_corrcoef as spearman

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training', 'MVTM'))

from load_model import load_model_from_huggingface, load_model_from_local
from crc_orion_channel_info import get_channel_info
from intensity_tokenized import get_intensities, get_spearman
from helper import get_dataloader


def main():
    parser = argparse.ArgumentParser(description='Run inference with pre-trained MVTM model')
    parser.add_argument('--val-file', type=str, required=True,
                        help='Path to validation HDF5 file')
    parser.add_argument('--input-channels', type=str, default='17,6,11,13',
                        help='Comma-separated input channel indices (default: H&E,CD8a,PD-L1,CD163)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--dataset-size', type=int, default=10000,
                        help='Number of cells to process (default: 10000)')
    parser.add_argument('--output', type=str, default='inference_results.csv',
                        help='Output CSV filename (default: inference_results.csv)')

    # Model source options
    source_group = parser.add_argument_group('Model source (HuggingFace or local)')
    source_group.add_argument('--hf-repo', type=str, default='changlab/miniMTI-CRC',
                              help='HuggingFace repo ID for model download')
    source_group.add_argument('--local-mvtm-ckpt', type=str, default=None,
                              help='Local path to MVTM checkpoint (skips HuggingFace download)')
    source_group.add_argument('--local-if-config', type=str, default=None,
                              help='Local path to IF VQGAN config')
    source_group.add_argument('--local-if-ckpt', type=str, default=None,
                              help='Local path to IF VQGAN checkpoint')
    source_group.add_argument('--local-he-config', type=str, default=None,
                              help='Local path to H&E VQGAN config')
    source_group.add_argument('--local-he-ckpt', type=str, default=None,
                              help='Local path to H&E VQGAN checkpoint')

    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load model
    print("\n--- Loading model ---")
    if args.local_mvtm_ckpt is not None:
        print("Loading from local paths...")
        model, tokenizer = load_model_from_local(
            mvtm_ckpt=args.local_mvtm_ckpt,
            if_config=args.local_if_config,
            if_ckpt=args.local_if_ckpt,
            he_config=args.local_he_config,
            he_ckpt=args.local_he_ckpt,
            device=device,
        )
    else:
        print(f"Downloading from HuggingFace: {args.hf_repo}")
        (model, tokenizer), config = load_model_from_huggingface(
            repo_id=args.hf_repo,
            device=device,
        )
    print("Model loaded successfully!")

    # Load data
    print("\n--- Loading data ---")
    data_config = {
        "downscale": False,
        "remove_he": False,
        "deconvolve_he": False,
        "rescale": True,
        "remove_background": False,
    }
    dataloader = get_dataloader(
        args.val_file, args.batch_size, args.dataset_size, **data_config
    )
    print(f"Loaded {len(dataloader.dataset)} cells")

    # Set up channels
    channels, _, _ = get_channel_info()
    channels.append('H&E')
    ch2stain = {i: ch for i, ch in enumerate(channels)}

    # Parse input channels
    input_ch_idx = [int(x) for x in args.input_channels.split(',')]
    input_markers = [channels[i] for i in input_ch_idx]
    predicted_markers = [channels[i] for i in range(len(channels)) if i not in input_ch_idx]

    print(f"\n--- Input panel ({len(input_ch_idx)} markers) ---")
    for idx, name in zip(input_ch_idx, input_markers):
        print(f"  [{idx}] {name}")

    print(f"\n--- Predicting {len(predicted_markers)} markers ---")
    for name in predicted_markers:
        print(f"  {name}")

    # Run inference
    print("\n--- Running inference ---")
    mints, pmints, ssims, logits = get_intensities(
        model=model,
        tokenizer=tokenizer,
        panel=input_ch_idx,
        val_loader=dataloader,
        ch2stain=ch2stain,
        calculate_ssims=False,
        device=device,
        return_meta=False,
        T=1,
    )

    # Calculate Spearman correlations
    print("\n--- Results ---")
    corrs = get_spearman(mints, pmints)

    masked_ch_idx = [i for i in range(len(channels)) if i not in input_ch_idx]
    print(f"\n{'Marker':<15} {'Spearman r':>10}")
    print("-" * 27)
    for i, ch_idx in enumerate(masked_ch_idx):
        marker_name = channels[ch_idx]
        corr_val = corrs[i].item() if corrs.dim() > 0 else corrs.item()
        print(f"{marker_name:<15} {corr_val:>10.4f}")

    mean_corr = corrs.mean().item()
    print("-" * 27)
    print(f"{'Mean':<15} {mean_corr:>10.4f}")

    # Save results
    data = np.concatenate([mints.numpy(), pmints.numpy()], axis=1)
    col_names = (
        [channels[i] for i in masked_ch_idx] +
        [channels[i] + '_imputed' for i in masked_ch_idx]
    )
    df = pd.DataFrame(data, columns=col_names)
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
