import os
import sys
import gc
import argparse
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch.nn.functional as F

from helper import get_model_and_tokenizer, get_dataloader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))


def calculate_pll_for_channel(model, tokenizer, dataloader, input_ch_idx, target_ch_idx,
                               predicted_tokens_file, device, channels, use_gt_only=False):
    """
    Calculate pseudo-log-likelihood for a single channel using per-token masking.

    Args:
        model: The MVTM model
        tokenizer: The VQGAN tokenizer
        dataloader: DataLoader for the original images
        input_ch_idx: List of input channel indices that were visible during original prediction
        target_ch_idx: Single channel index to calculate PLL for
        predicted_tokens_file: Path to saved token IDs (.npy file)
        device: torch device
        channels: List of channel names
        use_gt_only: If True, use GT tokens for all channels. Default False.

    Returns:
        pll_scores: Tensor of shape [N] with PLL score per patch
        metadata: List of patch metadata
    """
    print(f"Loading predicted tokens from {predicted_tokens_file}")
    predicted_tokens = torch.from_numpy(np.load(predicted_tokens_file)).to(device)

    expected_length = model.num_markers * model.max_positions
    assert predicted_tokens.shape[1] == expected_length, \
        f"Expected {expected_length} tokens, got {predicted_tokens.shape[1]}"

    num_patches = len(dataloader.dataset)
    pll_scores = torch.zeros(num_patches, device='cpu')
    metadata = []

    type_ids, position_ids = model._get_position_type_ids(device, channel_indices=None)

    is_input_channel = target_ch_idx in input_ch_idx
    channel_type = "input" if is_input_channel else "output"
    mode_str = "GT-only (all channels)" if use_gt_only else "hybrid (GT for inputs, predicted for outputs)"

    print(f"Calculating PLL for channel {target_ch_idx} ({channels[target_ch_idx]}) - {channel_type} channel")
    print(f"Mode: {mode_str}")
    print(f"Processing {num_patches} patches...")

    batch_size = None
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Ch {target_ch_idx}")):
        ims, masks, filepaths = batch
        gt = ims.to(device)

        if batch_size is None:
            batch_size = len(gt)

        actual_batch_size = gt.shape[0]

        with torch.no_grad():
            gt_tokens = tokenizer.tokenize(gt)

            batch_start = batch_idx * batch_size
            batch_end = batch_start + actual_batch_size
            pred_tokens = predicted_tokens[batch_start:batch_end].to(device)

            if use_gt_only:
                hybrid_tokens = gt_tokens.clone()
            else:
                hybrid_tokens = pred_tokens.clone()
                for ch_idx in input_ch_idx:
                    ch_start = model.max_positions * ch_idx
                    ch_end = model.max_positions * (ch_idx + 1)
                    hybrid_tokens[:, ch_start:ch_end] = gt_tokens[:, ch_start:ch_end]

                if is_input_channel:
                    ch_start = model.max_positions * target_ch_idx
                    ch_end = model.max_positions * (target_ch_idx + 1)
                    hybrid_tokens[:, ch_start:ch_end] = gt_tokens[:, ch_start:ch_end]

            batch_pll = torch.zeros(actual_batch_size, device=device)
            channel_start = model.max_positions * target_ch_idx

            for token_pos in range(model.max_positions):
                input_ids = hybrid_tokens.clone()
                token_idx = channel_start + token_pos
                input_ids[:, token_idx] = model.mask_id

                labels = torch.ones_like(input_ids) * -100
                labels[:, token_idx] = hybrid_tokens[:, token_idx]

                output = model.mvtm(
                    input_ids=input_ids,
                    token_type_ids=type_ids,
                    position_ids=position_ids,
                    labels=labels,
                    output_attentions=False,
                    output_hidden_states=False
                )

                logits = output['logits'][:, token_idx, :]
                target_token = hybrid_tokens[:, token_idx]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_prob = log_probs.gather(dim=1, index=target_token.unsqueeze(-1)).squeeze(-1)
                batch_pll += token_log_prob

                del input_ids, labels, output, logits, log_probs, token_log_prob

            pll_scores[batch_start:batch_end] = batch_pll.cpu()
            metadata.extend(filepaths)

        del batch, gt, gt_tokens, pred_tokens, hybrid_tokens, batch_pll
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return pll_scores, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate pseudo-log-likelihood from token IDs')
    parser.add_argument('sample_id', type=str, help='Sample ID (e.g., 10 for CRC10)')
    parser.add_argument('input_ch_idx', type=str, help='Comma-separated input channel indices (e.g., 17,6,11,13)')
    parser.add_argument('target_ch_idx', type=int, help='Target channel index to calculate PLL for')
    parser.add_argument('token_ids_file', type=str, help='Path to saved token IDs (.npy file)')
    parser.add_argument('--val-file', type=str, required=True, help='Path to validation HDF5 file')
    parser.add_argument('--config', type=str, default='configs/params_mvtm-256-he-rgb-tokenized.json', help='Path to config JSON')
    parser.add_argument('--use-gt-only', action='store_true', help='Use GT tokens for all channels')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--dataset-size', type=int, default=10_000_000, help='Max dataset size')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')

    with open(args.config, "r") as f:
        configs = json.load(f)

    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(configs, device)

    sid = f'CRC{args.sample_id}'
    input_ch_idx = [int(x) for x in args.input_ch_idx.split(',')]

    dataloader = get_dataloader(args.val_file, args.batch_size, args.dataset_size, **configs['data'])

    from crc_orion_channel_info import get_channel_info
    channels, _, _ = get_channel_info()
    if configs['data']['deconvolve_he']:
        channels.extend(['Hematoxylin', 'Eosin'])
    else:
        channels.append('H&E')

    pll_scores, metadata = calculate_pll_for_channel(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        input_ch_idx=input_ch_idx,
        target_ch_idx=args.target_ch_idx,
        predicted_tokens_file=args.token_ids_file,
        device=device,
        channels=channels,
        use_gt_only=args.use_gt_only
    )

    def get_xy(meta_):
        x = str(meta_).split('-')[-2]
        y = str(meta_).split('-')[-1]
        x = int(x.split('=')[-1])
        y = int(y.split('=')[-1].replace("'", ""))
        return x, y

    coords = np.vectorize(get_xy)(metadata)
    channel_name = channels[args.target_ch_idx]
    df = pd.DataFrame({
        'X_centroid': coords[0],
        'Y_centroid': coords[1],
        f'{channel_name}_pll': pll_scores.numpy()
    })

    if args.use_gt_only:
        mode_suffix = "gt-only"
    else:
        input_marker_names = "input-markers=" + "-".join(channels[ch] for ch in input_ch_idx)
        mode_suffix = input_marker_names

    save_fname = f'{sid}_pll_scores_{mode_suffix}_output={channel_name}.csv'
    df.to_csv(save_fname, index=False)
    print(f"\nSaved PLL scores to {save_fname}")
    print(f"Mean PLL: {pll_scores.mean().item():.4f}")
    print(f"Std PLL: {pll_scores.std().item():.4f}")

    del pll_scores, metadata, df
    gc.collect()
    torch.cuda.empty_cache()
