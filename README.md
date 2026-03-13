# miniMTI: minimal multiplex tissue imaging

**miniMTI** is a molecularly anchored virtual staining framework that determines the minimal set of experimentally measured markers required, alongside H&E, to accurately reconstruct large multiplex tissue imaging (MTI) panels while preserving biologically and clinically relevant information.

[[Paper]](https://www.biorxiv.org/content/10.64898/2026.01.21.700911v1) [[Model]](https://huggingface.co/changlab/miniMTI-CRC)

<p align="center">
  <img src="assets/miniMTI_fig1.png" alt="miniMTI overview" width="100%">
</p>

## Overview

miniMTI learns from paired same-section H&E-MTI data using a unified multimodal generative model that can condition on arbitrary combinations of measured marker channels, coupled with an iterative panel selection strategy to rank informative molecular anchors. The method works by:

1. **Tokenization**: Single-cell images (32x32 patches centered on segmented cells) are tokenized into discrete codes using VQGAN tokenizers — one for IF channels and one for H&E
2. **Masked Token Modeling**: A RoBERTa-based masked language model learns to predict masked channel tokens from visible channels, treating each marker as a "sentence" of 16 tokens
3. **Panel Selection**: An iterative greedy algorithm identifies which markers are most informative for predicting the rest, enabling rational panel reduction

The model is trained on the CRC-Orion dataset (colorectal cancer tissue whole slide images) with 17 IF markers + co-registered H&E.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone with submodules (includes the VQGAN tokenizer)
git clone --recurse-submodules https://github.com/ChangLab/miniMTI.git
cd miniMTI
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

```python
import torch
import h5py
from huggingface_hub import hf_hub_download
from eval.load_model import load_model_from_huggingface
from data.data import SingleCellDataset

# 1. Load pre-trained model and tokenizer from HuggingFace
(model, tokenizer), config = load_model_from_huggingface()
device = next(model.parameters()).device

# 2. Download example data (10k cells from CRC-Orion)
h5_path = hf_hub_download(
    "changlab/miniMTI-CRC-example", "example_CRC04_10k.h5", repo_type="dataset"
)

# 3. Load single-cell images from HDF5
with h5py.File(h5_path, "r") as f:
    dataset = SingleCellDataset(
        f["images"][:100], f["masks"][:100], f["metadata"][:100],
        remove_he=False, rescale=True, deconvolve_he=False,
    )
images = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)

# 4. Define input panel: H&E + CD8a + PD-L1 + CD163
input_channels = [17, 6, 11, 13]
masked_channels = [i for i in range(18) if i not in input_channels]

# 5. Tokenize, predict masked channels, and detokenize
with torch.no_grad():
    token_ids = tokenizer.tokenize(images)
    out, token_ids, labels, _ = model.predict(
        token_ids, masked_ch_idx=masked_channels, T=1, temp=0
    )
    predicted_images = tokenizer.detokenize(token_ids).reshape(images.shape)

# predicted_images now contains the full 18-channel panel
# Channels in input_channels are copied from the input;
# all other channels are predicted by the model
```

For a more complete example with evaluation metrics, see `scripts/inference_example.py`.

### Expected output

Running the Quick Start snippet on 100 cells from the example dataset with input panel H&E + CD8a + PD-L1 + CD163 produces the following Spearman correlations between real and predicted mean marker intensities:

| Predicted marker | Spearman *r* | | Predicted marker | Spearman *ρ* |
|------------------|-------------:|-|------------------|-------------:|
| DAPI             |       0.8524 | | CD3e             |       0.9470 |
| CD31             |       0.8051 | | E-cadherin       |       0.9410 |
| CD45             |       0.9741 | | Ki67             |       0.5163 |
| CD68             |       0.6889 | | PanCK            |       0.8511 |
| CD4              |       0.9326 | | aSMA             |       0.4230 |
| FOXP3            |       0.7263 | |                  |              |
| CD45RO           |       0.9278 | |                  |              |
| CD20             |       0.6621 | |                  |              |
| PD-L1            |       0.5180 | |                  |              |

Note: Correlations improve with larger evaluation sets (mean *ρ* = 0.80 at 10,000 cells). Results on 100 cells show higher variance due to the small sample size.

### Typical install time

On a standard desktop computer with a broadband internet connection (~100 Mbps):

| Step | Time |
|------|------|
| Clone repository + VQGAN submodule | ~1 min |
| Install dependencies (`uv sync` or `pip install`) | 5–10 min |
| Download pre-trained model weights from HuggingFace | 5–10 min |
| **Total** | **~15–20 min** |

Install time is dominated by downloading PyTorch (~2.5 GB) and the model weights (~5.3 GB).

### Runtime

Benchmarked on a single NVIDIA A40 (48 GB) with batch size 256:

| Step | Time |
|------|------|
| Model loading (from local cache) | ~5 min |
| Data loading (100 / 1k / 10k cells) | 0.2s / 0.4s / 3.3s |
| Inference (100 / 1k / 10k cells) | 11s / 93s / 15 min |
| Throughput | ~93 ms/cell |
| Peak GPU memory | ~21 GB |

First-time model download from HuggingFace (~5.3 GB) depends on network speed. Subsequent runs use the local cache.

## Data Format

Input data is stored in HDF5 files with the following structure:

| Dataset    | Shape           | Type    | Description |
|------------|-----------------|---------|-------------|
| `images`   | (N, 32, 32, 20) | uint8   | 17 IF channels + 3 H&E (RGB) channels |
| `masks`    | (N, 32, 32)     | bool    | Binary cell segmentation masks |
| `metadata` | (N,)            | string  | Cell IDs and coordinates: `<sample>-CellID-<id>-x=<x>-y=<y>` |

### CRC-Orion channel ordering (18 tokens)

| Index | Marker      | Index | Marker      |
|-------|-------------|-------|-------------|
| 0     | DAPI        | 9     | PD-L1       |
| 1     | CD31        | 10    | CD3e        |
| 2     | CD45        | 11    | CD163       |
| 3     | CD68        | 12    | E-cadherin  |
| 4     | CD4         | 13    | PD-1        |
| 5     | FOXP3       | 14    | Ki67        |
| 6     | CD8a        | 15    | PanCK       |
| 7     | CD45RO      | 16    | aSMA        |
| 8     | CD20        | 17    | H&E (RGB)   |

Each IF marker (indices 0–16) is encoded as a single-channel image by the VQGAN into 16 tokens. H&E (index 17) is encoded as a 3-channel RGB image into 16 tokens.

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

Pre-trained model weights and example data are hosted on HuggingFace:
- **Model**: [changlab/miniMTI-CRC](https://huggingface.co/changlab/miniMTI-CRC) — MVTM checkpoint + VQGAN tokenizers
- **Example data**: [changlab/miniMTI-CRC-example](https://huggingface.co/datasets/changlab/miniMTI-CRC-example) — 10k single-cell patches from CRC-Orion

### Requesting access

The model weights are gated for non-commercial academic use. To gain access:

1. Create a [HuggingFace](https://huggingface.co) account (use your institutional email)
2. Visit [changlab/miniMTI-CRC](https://huggingface.co/changlab/miniMTI-CRC) and click **"Agree and access repository"**
3. Fill in your name, affiliation, and agree to the non-commercial license

Access is typically granted within 24 hours. Once approved, authenticate locally:

```bash
# Install the HuggingFace CLI (included in project dependencies)
pip install huggingface-hub

# Log in with your HuggingFace token
huggingface-cli login
```

### Downloading model weights

```python
from eval.load_model import load_model_from_huggingface

# Downloads and caches all weights (~5.3 GB total), then loads the model
model, tokenizer, config = load_model_from_huggingface()
```

Or download individual files manually:

```python
from huggingface_hub import hf_hub_download

# MVTM masked token model (~3.4 GB)
mvtm_ckpt = hf_hub_download("changlab/miniMTI-CRC", "mvtm_model.ckpt")

# VQGAN IF tokenizer (~955 MB)
if_config = hf_hub_download("changlab/miniMTI-CRC", "tokenizer/if_config.yaml")
if_ckpt = hf_hub_download("changlab/miniMTI-CRC", "tokenizer/if_model.ckpt")

# VQGAN H&E tokenizer (~955 MB)
he_config = hf_hub_download("changlab/miniMTI-CRC", "tokenizer/he_config.yaml")
he_ckpt = hf_hub_download("changlab/miniMTI-CRC", "tokenizer/he_model.ckpt")
```

### Downloading example data

The example dataset does not require access approval:

```python
from huggingface_hub import hf_hub_download

# 10,000 single-cell patches from CRC-Orion sample CRC04 (~178 MB)
example_h5 = hf_hub_download(
    "changlab/miniMTI-CRC-example",
    "example_CRC04_10k.h5",
    repo_type="dataset",
)
```

### Architecture

| Component | Details |
|-----------|---------|
| Backbone  | RoBERTa (24 layers, 16 heads, dim=1024) |
| IF Tokenizer | VQGAN (codebook=256, latent=4x4) |
| H&E Tokenizer | VQGAN (codebook=256, latent=4x4) |
| Sequence length | 18 markers x 16 tokens = 288 tokens |
| Training | Masked token prediction with cosine masking schedule |

## Citation

If you use miniMTI in your research, please cite our preprint:

```bibtex
@article{sims2026minimti,
  title={miniMTI: minimal multiplex tissue imaging enhances biomarker expression prediction from histology},
  author={Sims, Z. and Govindarajan, S. and Ait-Ahmad, K. and Ak, C. and Kuykendall, M. and Mills, G. B. and Eksi, E. and Chang, Y. H.},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.01.21.700911},
  url={https://www.biorxiv.org/content/10.64898/2026.01.21.700911v1}
}
```

## License

This project is licensed for non-commercial academic and research use only. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was carried out with major support from the National Cancer Institute (NCI) Human Tumor Atlas Network (HTAN) Research Centers at OHSU (U2CCA233280), and funding from the Cancer Early Detection Advanced Research Center at Oregon Health & Science University, Knight Cancer Institute (CEDAR 2023-1796). Y.H.C. is supported by R01 CA253860, R01 CA276224, U01 294548, and the Kuni Foundation Imagination Grants. S.E.E. is supported by Prostate Cancer Foundation (GCNCR1865A) and International Alliance for Cancer Early Detection (ACED 2022-1508). The research reported in this publication used computational infrastructure supported by the Office of Research Infrastructure Programs, Office of the Director, National Institutes of Health, under Award Number S10OD034224. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
