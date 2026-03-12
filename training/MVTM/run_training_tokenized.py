import os
import sys
import argparse
import gc
import torch
import numpy as np
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from mvtm_tokenized import Tokenized_MVTM
from tokenized_data import get_tokenized_dataloader
from pytorch_lightning.utilities import rank_zero_only

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, some memory monitoring features will be disabled")

# Set random seed for reproducibility
pl.seed_everything(42, workers=True)


class MemoryMonitorCallback(Callback):
    """Monitors GPU memory usage and performs cleanup when usage is high."""

    def __init__(self, log_interval=100, memory_cleanup_interval=10,
                 memory_warning_threshold=0.95):
        super().__init__()
        self.log_interval = log_interval
        self.memory_cleanup_interval = memory_cleanup_interval
        self.memory_warning_threshold = memory_warning_threshold
        self.gpu_memory_history = []
        self.enable_aggressive_cleanup = False

    @rank_zero_only
    def _get_memory_stats(self):
        stats = {}
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            cpu_info = process.memory_info()
            stats["rss_mb"] = cpu_info.rss / (1024 * 1024)

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                max_mem = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                stats[f"gpu{i}_allocated_mb"] = allocated
                stats[f"gpu{i}_utilization"] = allocated / max_mem

                if len(self.gpu_memory_history) < 50:
                    self.gpu_memory_history.append(allocated / max_mem)
                else:
                    self.gpu_memory_history.pop(0)
                    self.gpu_memory_history.append(allocated / max_mem)

        return stats

    def _force_cleanup(self, trainer=None):
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if trainer is not None and hasattr(trainer, "model") and hasattr(trainer.model, "clear_cache"):
            trainer.model.clear_cache()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval == 0 and trainer.global_rank == 0:
            memory_stats = self._get_memory_stats()
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log(memory_stats)

            # Check memory pressure
            if len(self.gpu_memory_history) >= 5:
                recent_mem = np.array(self.gpu_memory_history[-5:])
                if np.mean(recent_mem) > self.memory_warning_threshold:
                    self._force_cleanup(trainer)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        self._force_cleanup(trainer)


# Argument parsing
parser = argparse.ArgumentParser(description='Run training with pre-tokenized MVTM')
parser.add_argument('--train-file', type=str, required=True, help='Path to the tokenized training dataset file')
parser.add_argument('--num-gpus', type=int, required=True, help="number of GPUs to use")
parser.add_argument('--batch-size', type=int, default=16, help="batch size for training")
parser.add_argument('--num-workers', type=int, default=4, help="number of workers for data loading")
parser.add_argument('--num-epochs', type=int, default=10, help="number of epochs to train")
parser.add_argument('--gradient-accumulation', type=int, default=1, help="number of batches to accumulate gradients for")
parser.add_argument('--precision', type=str, default='32', help="Precision for training (16, 32, bf16)")
parser.add_argument('--checkpoint-every', type=int, default=1000, help="Save checkpoints every N steps")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
parser.add_argument('--default-root-dir', type=str, default=None, help="Root directory for checkpoints and logs (defaults to current directory)")
args = parser.parse_args()


def train_model(train_file, num_gpus, batch_size, num_workers, num_epochs,
               gradient_accumulation=1, precision='32', checkpoint_every=1000,
               default_root_dir=None):

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    print(f"Loading data with batch size {batch_size} and {num_workers} workers...")
    train_loader = get_tokenized_dataloader(train_file, batch_size, num_workers)

    params = dict(
        lr=3e-6,
        weight_decay=0,
        num_markers=18,
        num_layers=24,
        num_heads=16,
        latent_dim=1024,
        num_codes=256,
        vq_dim=4,
        full_channel_mask=False,
        if_only_mask=True
    )

    wandb_logger = WandbLogger(project="MVTM-panel-reduction-tokenized", entity='changlab', resume='allow', log_model=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        mode="min",
        every_n_train_steps=checkpoint_every,
        save_top_k=3
    )

    memory_monitor = MemoryMonitorCallback(
        log_interval=50,
        memory_cleanup_interval=10,
        memory_warning_threshold=0.95
    )

    # Handle precision
    if precision == '32':
        precision_value = 32
    elif precision == '16':
        precision_value = 16
    elif precision == 'bf16':
        precision_value = 16
        print("Note: bf16 precision mapped to 16 for compatibility")
    else:
        precision_value = 32

    # Check for checkpoint to resume from
    checkpoint_path = None
    wandb_run_id = os.environ.get('WANDB_RUN_ID', None)
    if wandb_run_id:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(base_dir, "MVTM-panel-reduction-tokenized", wandb_run_id, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')],
                               key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
                               reverse=True)
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Found checkpoint to resume from: {checkpoint_path}")

    if checkpoint_path:
        model = Tokenized_MVTM(**params).load_from_checkpoint(checkpoint_path, strict=False, **params)
    else:
        model = Tokenized_MVTM(**params)
    model.batch_size = batch_size

    if default_root_dir is None:
        default_root_dir = os.path.dirname(os.path.abspath(__file__))

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, memory_monitor],
        max_epochs=num_epochs,
        strategy='ddp',
        default_root_dir=default_root_dir,
        accumulate_grad_batches=gradient_accumulation,
        precision=precision_value,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        detect_anomaly=False,
    )

    if trainer.global_rank == 0:
        params.update({
            'batch_size': batch_size,
            'num_workers': num_workers,
            'gradient_accumulation': gradient_accumulation,
            'precision': precision,
            'effective_batch_size': batch_size * num_gpus * gradient_accumulation,
        })
        wandb_logger.experiment.config.update(params, allow_val_change=True)

    wandb_logger.watch(model, log="all", log_freq=1000)

    print(f"Starting training with batch size {batch_size}, accumulation {gradient_accumulation}...")
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    if args.seed != 42:
        pl.seed_everything(args.seed, workers=True)

    train_model(
        train_file=args.train_file,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        gradient_accumulation=args.gradient_accumulation,
        precision=args.precision,
        checkpoint_every=args.checkpoint_every,
        default_root_dir=args.default_root_dir
    )
