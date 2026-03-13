"""
Utility for downloading and loading pre-trained MVTM model + VQGAN tokenizer
from HuggingFace Hub or from local paths.
"""
import os
import sys
import json
import torch

HF_REPO_ID = "changlab/cycif-panel-reduction"


def download_from_huggingface(repo_id=HF_REPO_ID, cache_dir=None):
    """Download model checkpoint and tokenizer weights from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (default: changlab/cycif-panel-reduction)
        cache_dir: Local cache directory (default: HuggingFace default cache)

    Returns:
        dict with local paths to downloaded files
    """
    from huggingface_hub import hf_hub_download

    files = {
        "mvtm_checkpoint": "mvtm_model.ckpt",
        "if_config": "tokenizer/if_config.yaml",
        "if_checkpoint": "tokenizer/if_model.ckpt",
        "he_config": "tokenizer/he_config.yaml",
        "he_checkpoint": "tokenizer/he_model.ckpt",
        "config": "config.json",
    }

    local_paths = {}
    for key, filename in files.items():
        print(f"Downloading {filename}...")
        local_paths[key] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )

    return local_paths


def load_model_from_huggingface(repo_id=HF_REPO_ID, device=None, cache_dir=None):
    """Download and load model + tokenizer from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        device: torch device (default: cuda if available, else cpu)
        cache_dir: Local cache directory for downloaded files

    Returns:
        model: Loaded Tokenized_MVTM model in eval mode
        tokenizer: Loaded Tokenizer with VQGAN weights
        config: Model configuration dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paths = download_from_huggingface(repo_id, cache_dir)

    with open(paths["config"], "r") as f:
        config = json.load(f)

    return _load_from_paths(
        mvtm_ckpt=paths["mvtm_checkpoint"],
        if_config=paths["if_config"],
        if_ckpt=paths["if_checkpoint"],
        he_config=paths["he_config"],
        he_ckpt=paths["he_checkpoint"],
        model_config=config["model"],
        tokenizer_config=config["tokenizer"],
        device=device,
    ), config


def load_model_from_local(mvtm_ckpt, if_config, if_ckpt, he_config, he_ckpt,
                          config_path=None, device=None):
    """Load model + tokenizer from local file paths.

    Args:
        mvtm_ckpt: Path to MVTM model checkpoint
        if_config: Path to IF VQGAN config YAML
        if_ckpt: Path to IF VQGAN checkpoint
        he_config: Path to H&E VQGAN config YAML
        he_ckpt: Path to H&E VQGAN checkpoint
        config_path: Optional path to config.json with model/tokenizer params
        device: torch device

    Returns:
        model: Loaded Tokenized_MVTM model in eval mode
        tokenizer: Loaded Tokenizer with VQGAN weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config_path is not None:
        with open(config_path, "r") as f:
            config = json.load(f)
        model_config = config["model"]
        tokenizer_config = config["tokenizer"]
    else:
        # Default model configuration for CRC-Orion
        model_config = {
            "num_markers": 18,
            "num_layers": 24,
            "num_heads": 16,
            "latent_dim": 1024,
            "num_codes": 256,
            "vq_dim": 4
        }
        tokenizer_config = {
            "vq_dim": 4,
            "vq_f_dim": 256,
            "num_if_channels": 17,
            "num_codes": 256,
        }

    # Override paths with provided local paths
    tokenizer_config["if_config_path"] = if_config
    tokenizer_config["if_ckpt_path"] = if_ckpt
    tokenizer_config["he_config_path"] = he_config
    tokenizer_config["he_ckpt_path"] = he_ckpt

    return _load_from_paths(
        mvtm_ckpt=mvtm_ckpt,
        if_config=if_config,
        if_ckpt=if_ckpt,
        he_config=he_config,
        he_ckpt=he_ckpt,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        device=device,
    )


def _load_from_paths(mvtm_ckpt, if_config, if_ckpt, he_config, he_ckpt,
                     model_config, tokenizer_config, device):
    """Internal helper to load model and tokenizer from resolved paths."""
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training', 'MVTM'))
    from tokenizer import Tokenizer
    from mvtm_tokenized import Tokenized_MVTM

    # Build tokenizer config with resolved paths
    tok_params = {
        "vq_dim": tokenizer_config["vq_dim"],
        "vq_f_dim": tokenizer_config["vq_f_dim"],
        "num_if_channels": tokenizer_config["num_if_channels"],
        "num_codes": tokenizer_config["num_codes"],
        "if_config_path": if_config,
        "if_ckpt_path": if_ckpt,
        "he_config_path": he_config,
        "he_ckpt_path": he_ckpt,
    }

    tokenizer = Tokenizer(**tok_params)
    tokenizer.if_tokenizer = tokenizer.if_tokenizer.to(device)
    if he_config:
        tokenizer.he_tokenizer = tokenizer.he_tokenizer.to(device)

    model = Tokenized_MVTM.load_from_checkpoint(mvtm_ckpt, strict=False, **model_config)
    model = model.to(device).eval()

    return model, tokenizer
