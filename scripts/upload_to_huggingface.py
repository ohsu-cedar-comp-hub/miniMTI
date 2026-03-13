"""
Upload miniMTI model weights and VQGAN tokenizer to HuggingFace Hub.

Usage:
    huggingface-cli login
    python scripts/upload_to_huggingface.py --repo-id changlab/miniMTI-CRC

By default, uses the checkpoint paths on the OHSU cluster. Override with
--mvtm-ckpt, --if-config, --if-ckpt, --he-config, --he-ckpt flags.

Requires: pip install huggingface_hub
"""
import os
import json
import argparse
from huggingface_hub import HfApi, create_repo

# Default source paths (OHSU cluster)
DEFAULT_MVTM_CKPT = "/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction-tokenized/hhdnp2ma/checkpoints/epoch=1-step=343000.ckpt"
DEFAULT_IF_CONFIG = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/configs/custom_vqgan_if_only_256.yaml"
DEFAULT_IF_CKPT = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/vqgan_checkpoints_opt/if_only_he_rgb_vqgan_models/if_only_256_ckpts/last.ckpt"
DEFAULT_HE_CONFIG = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/configs/custom_vqgan_he_rgb_256.yaml"
DEFAULT_HE_CKPT = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/taming-transformers/vqgan_checkpoints_opt/if_only_he_rgb_vqgan_models/he_rgb_256_ckpts/last.ckpt"


def main():
    parser = argparse.ArgumentParser(description='Upload miniMTI weights to HuggingFace Hub')
    parser.add_argument('--mvtm-ckpt', type=str, default=DEFAULT_MVTM_CKPT,
                        help='Path to MVTM model checkpoint')
    parser.add_argument('--if-config', type=str, default=DEFAULT_IF_CONFIG,
                        help='Path to IF VQGAN config YAML')
    parser.add_argument('--if-ckpt', type=str, default=DEFAULT_IF_CKPT,
                        help='Path to IF VQGAN checkpoint')
    parser.add_argument('--he-config', type=str, default=DEFAULT_HE_CONFIG,
                        help='Path to H&E VQGAN config YAML')
    parser.add_argument('--he-ckpt', type=str, default=DEFAULT_HE_CKPT,
                        help='Path to H&E VQGAN checkpoint')
    parser.add_argument('--repo-id', type=str, default='changlab/miniMTI-CRC',
                        help='HuggingFace repo ID')
    parser.add_argument('--private', action='store_true', help='Make the repository private')
    args = parser.parse_args()

    api = HfApi()

    # Create the repository if it doesn't exist
    try:
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"Repository {args.repo_id} ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Verify all source files exist
    uploads = [
        (args.mvtm_ckpt, "mvtm_model.ckpt"),
        (args.if_config, "tokenizer/if_config.yaml"),
        (args.if_ckpt, "tokenizer/if_model.ckpt"),
        (args.he_config, "tokenizer/he_config.yaml"),
        (args.he_ckpt, "tokenizer/he_model.ckpt"),
    ]

    for src, dst in uploads:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"  {dst}: {size_mb:.1f} MB <- {src}")

    # Model config
    config = {
        "model_id": "hhdnp2ma",
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

    config_path = "/tmp/minimti_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Model card
    model_card = """---
license: other
license_name: ohsu-non-commercial
license_link: https://github.com/ChangLab/cycif-panel-reduction/blob/publication/LICENSE
tags:
  - biology
  - multiplex-imaging
  - virtual-staining
  - computational-pathology
  - cycif
---

# miniMTI: minimal multiplex tissue imaging

Pre-trained model weights for **miniMTI**, a molecularly anchored virtual staining
framework that determines the minimal set of experimentally measured markers required,
alongside H&E, to accurately reconstruct large multiplex tissue imaging (MTI) panels.

**Paper:** [bioRxiv 2026.01.21.700911](https://www.biorxiv.org/content/10.64898/2026.01.21.700911v1)
**Code:** [GitHub](https://github.com/ChangLab/cycif-panel-reduction)

## Model Architecture

| Component | Details |
|-----------|---------|
| Backbone  | RoBERTa (24 layers, 16 heads, dim=1024) |
| IF Tokenizer | VQGAN (codebook=256, latent=4x4) |
| H&E Tokenizer | VQGAN (codebook=256, latent=4x4) |
| Sequence length | 18 channels x 16 tokens = 288 tokens |
| Training | Masked token prediction with cosine masking schedule |

## Files

- `mvtm_model.ckpt` — MVTM masked token model checkpoint (3.4 GB)
- `tokenizer/if_config.yaml` — IF VQGAN configuration
- `tokenizer/if_model.ckpt` — IF VQGAN checkpoint (955 MB)
- `tokenizer/he_config.yaml` — H&E VQGAN configuration
- `tokenizer/he_model.ckpt` — H&E VQGAN checkpoint (955 MB)
- `config.json` — Model and tokenizer configuration

## Usage

```python
from eval.load_model import load_model_from_huggingface

model, tokenizer = load_model_from_huggingface()
```

See the [repository](https://github.com/ChangLab/cycif-panel-reduction)
for full documentation.

## Citation

```bibtex
@article{sims2026minimti,
  title={miniMTI: minimal multiplex tissue imaging enhances biomarker expression prediction from histology},
  author={Sims, Z. and Govindarajan, S. and Ait-Ahmad, K. and Ak, C. and Kuykendall, M. and Mills, G. B. and Eksi, E. and Chang, Y. H.},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.01.21.700911}
}
```
"""

    model_card_path = "/tmp/minimti_README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)

    # Upload small files first
    print(f"\nUploading to {args.repo_id}...")

    for path, name in [(config_path, "config.json"), (model_card_path, "README.md")]:
        api.upload_file(path_or_fileobj=path, path_in_repo=name,
                        repo_id=args.repo_id, repo_type="model")
        print(f"  Uploaded {name}")

    # Upload model files
    for src, dst in uploads:
        print(f"  Uploading {dst}...")
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id=args.repo_id, repo_type="model")
        print(f"  Uploaded {dst}")

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
