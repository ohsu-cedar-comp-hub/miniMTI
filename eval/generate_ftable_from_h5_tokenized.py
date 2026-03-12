import os
import sys
import gc
import torch
import json
import numpy as np
import pandas as pd
from intensity_tokenized import get_intensities, get_spearman
from helper import get_ckpt, get_model_and_tokenizer, get_dataloader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))


def generate_imputed_feature_table(model, tokenizer, dataloader, channels, unmasked_ch_idx, save_fname, device, T=1, return_logits=False, output_ch_idx=None, return_token_ids=False):
    """Generate imputed feature table.

    Args:
        unmasked_ch_idx: Input channel indices (kept visible)
        output_ch_idx: Optional output channel indices (to predict). If None, predicts all non-input channels.
        return_token_ids: If True, saves intermediate token IDs before detokenization
    """
    ch2stain = {i: ch for i, ch in enumerate(channels)}

    # Determine output channels
    if output_ch_idx is None:
        masked_ch_idx = [i for i, ch in enumerate(channels) if i not in unmasked_ch_idx]
    else:
        masked_ch_idx = output_ch_idx

    result = get_intensities(
        model=model,
        tokenizer=tokenizer,
        panel=unmasked_ch_idx,
        val_loader=dataloader,
        ch2stain=ch2stain,
        calculate_ssims=False,
        device=device,
        return_meta=True,
        return_logits=return_logits,
        T=T,
        output_panel=output_ch_idx,
        return_token_ids=return_token_ids
    )

    if return_token_ids:
        mints, pmints, ssims, meta, logits, token_ids = result
    else:
        mints, pmints, ssims, meta, logits = result

    def get_xy(meta_):
        x = str(meta_).split('-')[-2]
        y = str(meta_).split('-')[-1]
        x = int(x.split('=')[-1])
        y = int(y.split('=')[-1].replace("'", ""))
        return x, y

    coords = np.vectorize(get_xy)(meta)
    data = np.concatenate([mints, pmints, np.expand_dims(coords[0], 1), np.expand_dims(coords[1], 1)], axis=1)
    df = pd.DataFrame(data, columns=[ch for i, ch in enumerate(channels) if i in masked_ch_idx] + [ch + '_imputed' for i, ch in enumerate(channels) if i in masked_ch_idx] + ['X_centroid', 'Y_centroid'])

    if return_logits:
        logits_fname = save_fname.replace('.csv', 'logits.npy')
        np.save(logits_fname, logits.numpy())
        print(f"Saved logits to {logits_fname}")

    if return_token_ids:
        token_ids_fname = save_fname.replace('.csv', 'token_ids.npy')
        np.save(token_ids_fname, token_ids.numpy())
        print(f"Saved token IDs to {token_ids_fname}")

    df.to_csv(save_fname, index=False)
    print(f"Saved results to {save_fname}")

    to_delete = [mints, pmints, data, df, logits]
    if return_token_ids:
        to_delete.append(token_ids)
    del to_delete
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate imputed feature tables from tokenized MVTM')
    parser.add_argument('sample_id', type=str, help='Sample ID (e.g., 10 for CRC10)')
    parser.add_argument('--val-file', type=str, required=True, help='Path to validation HDF5 file')
    parser.add_argument('--config', type=str, default='configs/params_mvtm-256-he-rgb-tokenized.json', help='Path to config JSON')
    parser.add_argument('--input-channels', type=str, default='17,6,11,13', help='Comma-separated input channel indices')
    parser.add_argument('--output-channels', type=str, default=None, help='Comma-separated output channel indices (optional)')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--dataset-size', type=int, default=10_000_000, help='Max dataset size')
    parser.add_argument('--return-logits', action='store_true', help='Save logits')
    parser.add_argument('--return-token-ids', action='store_true', help='Save intermediate token IDs')
    parser.add_argument('--T', type=int, default=1, help='Number of decoding steps')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')

    with open(args.config, "r") as f:
        configs = json.load(f)

    model, tokenizer = get_model_and_tokenizer(configs, device)

    sid = f'CRC{args.sample_id}'
    dataloader = get_dataloader(args.val_file, args.batch_size, args.dataset_size, **configs['data'])

    from crc_orion_channel_info import get_channel_info
    channels, _, _ = get_channel_info()

    if configs['data']['deconvolve_he']:
        channels.extend(['Hematoxylin', 'Eosin'])
    else:
        channels.append('H&E')

    unmasked_ch_idx = [int(x) for x in args.input_channels.split(',')]

    output_ch_idx = None
    if args.output_channels:
        output_ch_idx = [int(x) for x in args.output_channels.split(',')]

    # Build marker names for filename
    input_marker_names = "input-markers=" + "-".join(channels[ch] for ch in unmasked_ch_idx)
    if output_ch_idx is not None:
        output_marker_names = "output-markers=" + "-".join(channels[ch] for ch in output_ch_idx)
        marker_description = f"{input_marker_names}_{output_marker_names}"
    else:
        marker_description = input_marker_names

    num_output = len(output_ch_idx) if output_ch_idx else len([i for i in range(len(channels)) if i not in unmasked_ch_idx])
    save_fname = f'{sid}_real_predicted_features_input={len(unmasked_ch_idx)}_output={num_output}_model_id={configs["model_id"]}_all_T={args.T}-{marker_description}.csv'

    generate_imputed_feature_table(
        model, tokenizer, dataloader, channels, unmasked_ch_idx,
        save_fname, device, args.T,
        return_logits=args.return_logits,
        output_ch_idx=output_ch_idx,
        return_token_ids=args.return_token_ids
    )
