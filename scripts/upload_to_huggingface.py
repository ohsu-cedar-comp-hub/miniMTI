"""
One-time script to upload model weights and tokenizer to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py \
        --mvtm-ckpt /path/to/mvtm_model.ckpt \
        --if-config /path/to/if_config.yaml \
        --if-ckpt /path/to/if_model.ckpt \
        --he-config /path/to/he_config.yaml \
        --he-ckpt /path/to/he_model.ckpt \
        --repo-id changlab/cycif-panel-reduction

Requires: pip install huggingface_hub
You must be logged in: huggingface-cli login
"""
import os
import json
import argparse
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description='Upload model weights to HuggingFace Hub')
    parser.add_argument('--mvtm-ckpt', type=str, required=True, help='Path to MVTM model checkpoint')
    parser.add_argument('--if-config', type=str, required=True, help='Path to IF VQGAN config YAML')
    parser.add_argument('--if-ckpt', type=str, required=True, help='Path to IF VQGAN checkpoint')
    parser.add_argument('--he-config', type=str, required=True, help='Path to H&E VQGAN config YAML')
    parser.add_argument('--he-ckpt', type=str, required=True, help='Path to H&E VQGAN checkpoint')
    parser.add_argument('--repo-id', type=str, default='changlab/cycif-panel-reduction',
                        help='HuggingFace repo ID (default: changlab/cycif-panel-reduction)')
    parser.add_argument('--private', action='store_true', help='Make the repository private')
    args = parser.parse_args()

    api = HfApi()

    # Create the repository if it doesn't exist
    try:
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"Repository {args.repo_id} ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Model architecture config
    config = {
        "model": {
            "num_markers": 18,
            "num_layers": 24,
            "num_heads": 16,
            "latent_dim": 1024,
            "num_codes": 256,
            "vq_dim": 4
        },
        "tokenizer": {
            "vq_dim": 4,
            "vq_f_dim": 256,
            "num_if_channels": 17,
            "num_codes": 256,
            "if_config_path": "tokenizer/if_config.yaml",
            "if_ckpt_path": "tokenizer/if_model.ckpt",
            "he_config_path": "tokenizer/he_config.yaml",
            "he_ckpt_path": "tokenizer/he_model.ckpt"
        },
        "data": {
            "downscale": False,
            "remove_he": False,
            "deconvolve_he": False,
            "rescale": True,
            "remove_background": False
        },
        "channels": [
            "DAPI", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a",
            "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin",
            "PD-1", "Ki67", "PanCK", "aSMA", "H&E"
        ]
    }

    # Save config locally then upload
    config_path = "/tmp/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Upload files
    uploads = [
        (args.mvtm_ckpt, "mvtm_model.ckpt"),
        (args.if_config, "tokenizer/if_config.yaml"),
        (args.if_ckpt, "tokenizer/if_model.ckpt"),
        (args.he_config, "tokenizer/he_config.yaml"),
        (args.he_ckpt, "tokenizer/he_model.ckpt"),
        (config_path, "config.json"),
    ]

    for local_path, repo_path in uploads:
        print(f"Uploading {local_path} -> {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"  Done.")

    # Upload model card
    model_card = """---
license: apache-2.0
tags:
  - biology
  - microscopy
  - cycif
  - panel-reduction
  - masked-language-model
---

# CycIF Panel Reduction with Masked Token Modeling (MVTM)

Pre-trained model for predicting missing immunofluorescence (IF) markers from
reduced antibody panels and co-registered H&E images.

## Model Description

This model uses a masked token modeling approach where:
1. Single-cell images are tokenized into discrete codes using separate VQGAN
   tokenizers for IF and H&E channels
2. A RoBERTa-based masked language model learns to predict masked channel tokens
   from visible channels

## Training Data

Trained on the CRC-Orion dataset (colorectal cancer tissue microarrays) with
17 IF markers + co-registered H&E.

## Usage

```python
from eval.load_model import load_model_from_huggingface

model, tokenizer = load_model_from_huggingface()
```

See the [repository](https://github.com/ChangLab/cycif-panel-reduction) for
full documentation.

## Citation

Please cite our Nature Methods paper (forthcoming).
"""

    model_card_path = "/tmp/README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)

    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"\nAll files uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
