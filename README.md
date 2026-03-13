# CycIF Panel Reduction with Masked Token Modeling

Predicting missing immunofluorescence (IF) markers from reduced antibody panels using masked token modeling (MVTM).

## Overview

This repository contains the code for our Nature Methods paper on CycIF panel reduction. The method works by:

1. **Tokenization**: Single-cell images (32x32 patches centered on segmented cells) are tokenized into discrete codes using VQGAN tokenizers — one for IF channels and one for H&E
2. **Masked Token Modeling**: A RoBERTa-based masked language model learns to predict masked channel tokens from visible channels, treating each marker as a "sentence" of 16 tokens
3. **Panel Selection**: An iterative greedy algorithm identifies which markers are most informative for predicting the rest, enabling rational panel reduction

The model is trained on the CRC-Orion dataset (colorectal cancer tissue microarrays) with 17 IF markers + co-registered H&E.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone with submodules (includes the VQGAN tokenizer)
git clone --recurse-submodules https://github.com/ChangLab/cycif-panel-reduction.git
cd cycif-panel-reduction
uv sync
```

If you already cloned without `--recurse-submodules`, initialize the VQGAN submodule:

```bash
git submodule update --init --recursive
```

Alternatively, install dependencies manually with pip:

```bash
pip install torch torchvision pytorch-lightning transformers einops omegaconf \
    h5py numpy pandas scikit-image scikit-learn scipy matplotlib seaborn \
    tqdm torchmetrics wandb huggingface-hub
```

### VQGAN Tokenizer

The VQGAN tokenizer ([taming-transformers](https://github.com/CompVis/taming-transformers)) is included as a git submodule under `third_party/taming-transformers/`. It is used for encoding single-cell images into discrete tokens and decoding predicted tokens back into images. The submodule is automatically added to the Python path when importing `training.MVTM.tokenizer`, with compatibility patches applied for PyTorch 2.0+.

Pre-trained VQGAN checkpoints (IF and H&E tokenizers) are available on HuggingFace alongside the main model weights.

## Quick Start

### Download pre-trained model and run inference

```python
from eval.load_model import load_model_from_huggingface

# Downloads model weights from HuggingFace Hub (cached locally)
model, tokenizer = load_model_from_huggingface()
```

### Run the example inference script

```bash
python scripts/inference_example.py \
    --val-file /path/to/sample.h5 \
    --input-channels 17,6,11,13 \
    --dataset-size 10000
```

This will:
- Load the pre-trained model from HuggingFace
- Run inference with H&E + CD8a + PD-L1 + CD163 as input
- Print Spearman correlations for each predicted marker
- Save a CSV with real vs. predicted mean intensities

## Data Format

Input data is stored in HDF5 files with the following structure:

| Dataset    | Shape           | Type    | Description |
|------------|-----------------|---------|-------------|
| `images`   | (N, 32, 32, 20) | uint8   | 17 IF channels + 3 H&E (RGB) channels |
| `masks`    | (N, 32, 32)     | bool    | Binary cell segmentation masks |
| `metadata` | (N,)            | string  | Cell IDs and coordinates: `<sample>-CellID-<id>-x=<x>-y=<y>` |

### CRC-Orion channel ordering (18 markers)

| Index | Marker      | Index | Marker      |
|-------|-------------|-------|-------------|
| 0     | DAPI        | 9     | CD45RO      |
| 1     | CD31        | 10    | CD20        |
| 2     | CD45        | 11    | PD-L1       |
| 3     | CD68        | 12    | CD3e        |
| 4     | CD4         | 13    | CD163       |
| 5     | FOXP3       | 14    | E-cadherin  |
| 6     | CD8a        | 15    | PD-1        |
| 7     | CD45RO      | 16    | Ki67        |
| 8     | CD20        | 17    | H&E (RGB)   |

Note: DAPI (index 0) and PanCK/aSMA are included in the IF channels. H&E is treated as a single multi-channel token.

## Full Pipeline

### 1. Data Preprocessing

Convert raw whole-slide images into single-cell HDF5 patches:

```bash
python data/process_orion_crc.py \
    --data-dir /path/to/Orion_CRC/ \
    --mask-dir /path/to/mesmer_masks/ \
    --save-dir /path/to/output/ \
    --sample-ids 01 02 03 04 05
```

### 2. Create Training Files

Merge individual sample HDF5 files into combined training/validation sets:

```bash
python data/create_training_files.py \
    --data-dir /path/to/h5_files/ \
    --val-samples data/val_samples_crc_orion.txt \
    --batch-name CRC05-06
```

### 3. Pre-tokenization

Convert image data into discrete tokens for efficient training:

```bash
python training/MVTM/pretokenize_data.py \
    --config-path-if /path/to/if_config.yaml \
    --ckpt-path-if /path/to/if_model.ckpt \
    --config-path-he /path/to/he_config.yaml \
    --ckpt-path-he /path/to/he_model.ckpt \
    --train-file /path/to/train.h5 \
    --val-file /path/to/val.h5 \
    --output-dir /path/to/tokenized/ \
    --num-channels 20
```

### 4. Training

Train the masked token model on pre-tokenized data:

```bash
python training/MVTM/run_training_tokenized.py \
    --train-file /path/to/tokenized/train_tokenized.h5 \
    --num-gpus 4 \
    --batch-size 16 \
    --num-epochs 10
```

### 5. Panel Selection

Run iterative panel selection to find the optimal marker ordering:

```bash
python eval/run_panel_selection_tokenized.py \
    --val-dataset /path/to/panel_selection_data.h5 \
    --val-dataset-size 5000 \
    --param-file eval/configs/params_mvtm-256-he-rgb-tokenized.json \
    --max-panel-size 18 \
    --gpu-id 0 \
    --batch-size 64
```

### 6. Evaluation

Generate feature tables with real and predicted mean intensities:

```bash
python eval/generate_ftable_from_h5_tokenized.py 10 \
    --val-file /path/to/CRC10.h5 \
    --config eval/configs/params_mvtm-256-he-rgb-tokenized.json \
    --input-channels 17,6,11,13
```

### 7. PLL Scoring

Calculate pseudo-log-likelihood scores for predicted channels:

```bash
python eval/calculate_pll_from_tokens.py 10 17,6,11,13 5 /path/to/token_ids.npy \
    --val-file /path/to/CRC10.h5 \
    --config eval/configs/params_mvtm-256-he-rgb-tokenized.json
```

## Pre-trained Models

Pre-trained model weights are hosted on HuggingFace:
- **Model**: [changlab/cycif-panel-reduction](https://huggingface.co/changlab/cycif-panel-reduction)
- **Example data**: [changlab/cycif-panel-reduction-example](https://huggingface.co/datasets/changlab/cycif-panel-reduction-example)

### Architecture

| Component | Details |
|-----------|---------|
| Backbone  | RoBERTa (24 layers, 16 heads, dim=1024) |
| IF Tokenizer | VQGAN (codebook=256, latent=4x4) |
| H&E Tokenizer | VQGAN (codebook=256, latent=4x4) |
| Sequence length | 18 channels x 16 tokens = 288 tokens |
| Training | Masked token prediction with cosine masking schedule |

## Citation

```bibtex
@article{simsz2026cycif,
  title={CycIF Panel Reduction with Masked Token Modeling},
  author={...},
  journal={Nature Methods},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported by grants from the National Institutes of Health (NIH), the Chan Zuckerberg Initiative (CZI), and the Stanford Data Science Initiative. We thank the members of the Chang Lab for their feedback and assistance throughout the development of this project.
