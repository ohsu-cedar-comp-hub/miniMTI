import os
import sys
import argparse
import gc
import torch
import time
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from mvtm_tokenized import Tokenized_MVTM
from tokenized_data import get_tokenized_dataloader
from pytorch_lightning.utilities import rank_zero_only

# Try to import psutil, but don't fail if it's not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, some memory monitoring features will be disabled")
    print("To install: pip install psutil")

# Set random seed for reproducibility
pl.seed_everything(69, workers=True)

# Advanced memory monitoring and management callback
class MemoryMonitorCallback(Callback):
    def __init__(self, log_interval=100, memory_cleanup_interval=10, 
                 memory_warning_threshold=0.95, memory_history_size=50):
        super().__init__()
        self.log_interval = log_interval
        self.memory_cleanup_interval = memory_cleanup_interval
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_history_size = memory_history_size
        
        # Keep track of memory usage history
        self.gpu_memory_history = []
        self.last_warning_time = 0
        self.enable_aggressive_cleanup = False
        self.batch_times = []
        
    @rank_zero_only
    def _get_memory_stats(self):
        """Get detailed memory statistics for CPU and GPU"""
        stats = {}
        
        # CPU memory stats with psutil if available
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            cpu_info = process.memory_info()
            stats["rss_mb"] = cpu_info.rss / (1024 * 1024)  # Resident Set Size
            stats["vms_mb"] = cpu_info.vms / (1024 * 1024)  # Virtual Memory Size
            
            # System memory
            system_mem = psutil.virtual_memory()
            stats["system_percent"] = system_mem.percent
            stats["system_available_gb"] = system_mem.available / (1024 * 1024 * 1024)
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)    # MB
                max_mem = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)  # MB
                
                stats[f"gpu{i}_allocated_mb"] = allocated
                stats[f"gpu{i}_reserved_mb"] = reserved
                stats[f"gpu{i}_utilization"] = allocated / max_mem
                stats[f"gpu{i}_max_mb"] = max_mem
                
                # Track the highest GPU usage for warning threshold
                if len(self.gpu_memory_history) < self.memory_history_size:
                    self.gpu_memory_history.append(allocated / max_mem)
                else:
                    self.gpu_memory_history.pop(0)
                    self.gpu_memory_history.append(allocated / max_mem)
        
        return stats
    
    @rank_zero_only
    def check_memory_pressure(self, trainer):
        """Check if memory pressure is too high and take action if needed"""
        # Only run this check if we have enough history
        if len(self.gpu_memory_history) < 5:
            return
            
        # Calculate mean and std of recent memory usage
        recent_mem = np.array(self.gpu_memory_history[-5:])
        mean_usage = np.mean(recent_mem)
        
        # Check if memory usage is approaching the threshold
        if mean_usage > self.memory_warning_threshold:
            current_time = time.time()
            
            # Only warn once every 5 minutes to avoid spam
            if current_time - self.last_warning_time > 300:
                print(f"⚠️ WARNING: High GPU memory usage detected ({mean_usage:.1%})")
                print("Enabling aggressive memory management...")
                self.last_warning_time = current_time
                self.enable_aggressive_cleanup = True
                
                # Force a full cleanup now
                self._force_cleanup(trainer)
    
    @rank_zero_only            
    def _force_cleanup(self, trainer=None):
        """Aggressively clean up memory"""
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
        # If we have a trainer and model, call the model's clear_cache method
        if trainer is not None and hasattr(trainer, "model") and hasattr(trainer.model, "clear_cache"):
            trainer.model.clear_cache()
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Record batch start time to measure throughput
        self.batch_start_time = time.time()
        
        # Clean up memory more frequently when in aggressive mode
        if self.enable_aggressive_cleanup and batch_idx % self.memory_cleanup_interval == 0:
            self._force_cleanup(trainer)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Calculate batch processing time
        if hasattr(self, 'batch_start_time'):
            batch_time = time.time() - self.batch_start_time
            if len(self.batch_times) < 50:
                self.batch_times.append(batch_time)
            else:
                self.batch_times.pop(0)
                self.batch_times.append(batch_time)
        
        # Log memory usage periodically
        if batch_idx % self.log_interval == 0 and trainer.global_rank == 0:
            # Get and log memory stats
            memory_stats = self._get_memory_stats()
            
            # Calculate average batch time
            if self.batch_times:
                memory_stats["avg_batch_time"] = sum(self.batch_times) / len(self.batch_times)
                memory_stats["samples_per_second"] = pl_module.batch_size / memory_stats["avg_batch_time"]
            
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log(memory_stats)
            
            # Check for high memory pressure
            self.check_memory_pressure(trainer)
            
            # Regular cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Clean up at the beginning of each epoch"""
        self._force_cleanup(trainer)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Clean up at the end of each epoch"""
        self._force_cleanup(trainer)
        
        # Reset aggressive cleanup flag at the end of each epoch
        self.enable_aggressive_cleanup = False
        
    def on_exception(self, trainer, pl_module, exception):
        """Cleanup on exception to prevent cascading failures"""
        print(f"⚠️ Exception detected: {str(exception)}")
        print("Performing emergency memory cleanup...")
        self._force_cleanup(trainer)

# Argument parsing for command line input
parser = argparse.ArgumentParser(description='Run training with pre-tokenized MVTM')
parser.add_argument('--train-file', type=str, required=True, help='Path to the tokenized training dataset file')
parser.add_argument('--num-gpus', type=int, required=True, help="number of GPUs to use")
parser.add_argument('--batch-size', type=int, default=16, help="batch size for training")
parser.add_argument('--num-workers', type=int, default=4, help="number of workers for data loading")
parser.add_argument('--num-epochs', type=int, default=10, help="number of epochs to train")
parser.add_argument('--gradient-accumulation', type=int, default=1, help="number of batches to accumulate gradients for")
parser.add_argument('--precision', type=str, default='32', help="Precision to use for training (16, 32, bf16)")
parser.add_argument('--find-batch-size', action='store_true', help="Automatically find the largest batch size that fits in memory")
parser.add_argument('--optimizer-state-mem-format', choices=['cpu', 'gpu'], default='gpu', 
                   help="Keep optimizer states on CPU to save GPU memory at the cost of slower training")
parser.add_argument('--use-checkpoint', action='store_true', 
                   help="Use gradient checkpointing to save memory at the cost of slower training")
parser.add_argument('--memory-threshold', type=float, default=0.95, 
                   help="Memory threshold for aggressive cleanup (0-1)")
parser.add_argument('--checkpoint-every', type=int, default=1000, 
                   help="Save checkpoints every N steps")
args = parser.parse_args()

def find_optimal_batch_size(train_file, num_gpus, start_batch_size=8, max_batch_size=128, num_workers=1, precision='32'):
    """
    Iteratively find the largest batch size that can fit in memory
    """
    print(f"🔍 Finding optimal batch size (starting at {start_batch_size}, max {max_batch_size})")
    
    batch_size = start_batch_size
    max_working_batch = None
    
    while batch_size <= max_batch_size:
        try:
            print(f"Testing batch size: {batch_size}")
            
            # Setup minimal model with the current batch size
            train_loader = get_tokenized_dataloader(train_file, batch_size, num_workers)
            
            # Create minimal model
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
            model = Tokenized_MVTM(**params)
            
            # Process a few batches to test memory usage
            model = model.to('cuda')
            for i, batch in enumerate(train_loader):
                if i >= 3:  # Just test a few batches
                    break
                    
                batch = batch.to('cuda')
                model.training_step(batch, i)
                
                # Clean up after each batch
                torch.cuda.empty_cache()
                gc.collect()
            
            # If we get here without OOM, record this batch size as successful
            max_working_batch = batch_size
            print(f"✅ Batch size {batch_size} works")
            
            # Try the next size
            batch_size *= 2
            
            # Clean up completely
            del model, train_loader
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)  # Give system time to free resources
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if 'out of memory' in str(e).lower():
                print(f"❌ Batch size {batch_size} caused OOM error")
                # If we get an OOM error, we need to clean up and try a smaller increment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                if max_working_batch:
                    # If we already found a working batch, try an intermediate size
                    next_batch = max_working_batch + (batch_size - max_working_batch) // 2
                    if next_batch == max_working_batch or next_batch == batch_size:
                        # We've converged to the highest working batch size
                        break
                    batch_size = next_batch
                else:
                    # If no batch size has worked yet, try half the current size
                    batch_size = batch_size // 2
                    if batch_size < start_batch_size:
                        raise ValueError("Could not find a working batch size. Consider reducing model size or using more aggressive memory optimizations.")
            else:
                # If the error is not OOM, re-raise it
                raise e
                
            # Wait a moment for resources to be fully released
            time.sleep(2)
    
    # Return the largest batch size that worked
    if max_working_batch:
        # Be conservative and use 90% of max to account for variation
        recommended = int(max_working_batch * 0.9)
        recommended = max(recommended, start_batch_size)  # Don't go below starting size
        print(f"🎯 Found optimal batch size: {max_working_batch}")
        print(f"🔒 Recommended conservative batch size: {recommended}")
        return recommended
    else:
        raise ValueError("Could not find a working batch size")

def train_model(train_file, num_gpus, batch_size, num_workers, num_epochs, 
               gradient_accumulation=1, precision='32', find_batch_size=False,
               optimizer_state_mem_format='gpu', use_checkpoint=False,
               memory_threshold=0.95, checkpoint_every=1000):
    
    # Optionally find the largest batch size that fits in memory
    if find_batch_size:
        batch_size = find_optimal_batch_size(train_file, num_gpus, start_batch_size=batch_size)
        print(f"Using auto-detected batch size: {batch_size}")
    
    # Set and display memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    if optimizer_state_mem_format == 'cpu':
        os.environ['PYTORCH_OPTIM_STATE_DICT_ON_CPU'] = '1'
        print("Storing optimizer states on CPU to save GPU memory")
    
    # Load pre-tokenized data
    print(f"Loading data with batch size {batch_size} and {num_workers} workers...")
    train_loader = get_tokenized_dataloader(train_file, batch_size, num_workers)
    
    # Setup model parameters with adjusted values for better memory efficiency
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
    
    # Setup wandb logging
    wandb_logger = WandbLogger(project="MVTM-panel-reduction-tokenized", entity='changlab', resume='allow', log_model=False)
    
    # Check for available checkpoint
    wandb_run_id = os.environ.get('WANDB_RUN_ID', None)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="loss", 
        mode="min", 
        every_n_train_steps=checkpoint_every,
        save_top_k=3  # Only keep the 3 best checkpoints to save disk space
    )
    
    # Add memory monitoring callback with custom threshold
    memory_monitor = MemoryMonitorCallback(
        log_interval=50,
        memory_cleanup_interval=10,
        memory_warning_threshold=memory_threshold,
        memory_history_size=50
    )
    
    # Convert precision to the format expected by PyTorch Lightning
    # Handle different PyTorch Lightning versions
    try:
        # Check PyTorch Lightning version
        import importlib
        pl_version = importlib.metadata.version('pytorch-lightning')
        print(f"PyTorch Lightning version: {pl_version}")
    except (ImportError, AttributeError):
        try:
            pl_version = pl.__version__
            print(f"PyTorch Lightning version: {pl_version}")
        except AttributeError:
            pl_version = "unknown"
            print("Could not determine PyTorch Lightning version")
    
    # Different versions have different precision formats
    if precision == '32':
        precision_value = 32
    elif precision == '16':
        precision_value = 16
    elif precision == 'bf16':
        # For older versions, bf16 might not be supported directly
        precision_value = 16
        print("Warning: bf16 precision requested but mapped to 16 for compatibility")
    else:
        # Default to 32 bit precision
        precision_value = 32
        print(f"Warning: Unknown precision '{precision}', defaulting to 32-bit")
    
    print(f"Using precision: {precision_value}")
    
    # Find checkpoint to resume from if WANDB_RUN_ID is set
    checkpoint_path = None
    if wandb_run_id:
        print(f"Looking for checkpoints for W&B run ID: {wandb_run_id}")
        # Use current directory as base path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(base_dir, "MVTM-panel-reduction-tokenized", wandb_run_id, "checkpoints")
        
        print(f"Looking in checkpoint directory: {checkpoint_dir}")
        if os.path.exists(checkpoint_dir):
            # Find the most recent checkpoint
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')], 
                               key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), 
                               reverse=True)
            
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Found checkpoint to resume from: {checkpoint_path}")
            else:
                print(f"No checkpoints found in {checkpoint_dir}")
        else:
            print(f"Checkpoint directory {checkpoint_dir} not found")
    
    # Add batch size to model to make it accessible in callbacks
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # Use strict=False to allow loading even if some parameters have changed
        model = Tokenized_MVTM(**params).load_from_checkpoint(checkpoint_path, strict=False, **params)
        model.batch_size = batch_size
        print(f"Model loaded with new batch_size: {batch_size}")
    else:
        print("Creating new model (no checkpoint found)")
        model = Tokenized_MVTM(**params)
        model.batch_size = batch_size
    
    # Setup additional trainer settings
    trainer_kwargs = {
        'accelerator': 'gpu',
        'devices': num_gpus,
        'logger': wandb_logger,
        'callbacks': [checkpoint_callback, memory_monitor],
        'max_epochs': num_epochs,
        'strategy': 'ddp',
        'default_root_dir': "/home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM",
        'accumulate_grad_batches': gradient_accumulation,  # Use gradient accumulation to reduce memory
        'precision': precision_value,  # Use lower precision if specified
        'gradient_clip_val': 1.0,  # Clip gradients to avoid exploding gradients
        'log_every_n_steps': 50,  # Log less frequently
        'detect_anomaly': False,  # Disable anomaly detection for speed
    }
    
    # Apply gradient checkpointing if enabled
    if use_checkpoint:
        # Enable gradient checkpointing in the transformer model - this reduces memory at the cost of compute time
        print("Enabling gradient checkpointing to save memory")
        # Different versions of transformers have different APIs for gradient checkpointing
        try:
            # Try newer API first
            model.mvtm.gradient_checkpointing_enable()
        except AttributeError:
            # Fall back to older API
            if hasattr(model.mvtm, 'config'):
                model.mvtm.config.use_cache = False  # Required for gradient checkpointing
                
                # For RobertaForMaskedLM, the attribute is on the encoder
                if hasattr(model.mvtm, 'roberta') and hasattr(model.mvtm.roberta, 'encoder'):
                    print("Enabling gradient checkpointing on roberta encoder")
                    model.mvtm.roberta.encoder.gradient_checkpointing = True
                # Generic fallback
                elif hasattr(model.mvtm, 'transformer'):
                    print("Enabling gradient checkpointing on transformer")
                    model.mvtm.transformer.gradient_checkpointing = True
                else:
                    print("WARNING: Could not enable gradient checkpointing - unsupported model structure")
            else:
                print("WARNING: Could not enable gradient checkpointing - config not found")
    
    # Setup trainer with memory-efficient settings
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Log configuration to wandb
    if trainer.global_rank == 0:
        params.update({
            'batch_size': batch_size,
            'num_workers': num_workers,
            'gradient_accumulation': gradient_accumulation,
            'precision': precision,
            'effective_batch_size': batch_size * num_gpus * gradient_accumulation,
            'optimizer_state_mem_format': optimizer_state_mem_format,
            'gradient_checkpointing': use_checkpoint,
            'memory_threshold': memory_threshold
        })
        wandb_logger.experiment.config.update(params, allow_val_change=True)
    
    # Watch model parameters but log less frequently to reduce overhead
    wandb_logger.watch(model, log="all", log_freq=1000)
    
    # Force complete cleanup before starting training
    print("Performing pre-training memory cleanup...")
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Train the model
    print(f"Starting training with batch size {batch_size}, accumulation {gradient_accumulation}...")
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    try:
        train_model(
            train_file=args.train_file, 
            num_gpus=args.num_gpus, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            num_epochs=args.num_epochs,
            gradient_accumulation=args.gradient_accumulation,
            precision=args.precision,
            find_batch_size=args.find_batch_size,
            optimizer_state_mem_format=args.optimizer_state_mem_format,
            use_checkpoint=args.use_checkpoint,
            memory_threshold=args.memory_threshold,
            checkpoint_every=args.checkpoint_every
        )
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        # Force cleanup on crash
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
