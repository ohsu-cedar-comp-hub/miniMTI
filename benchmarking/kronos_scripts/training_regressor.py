import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gc
import json
import os
import numpy as np
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import types
import h5py

# Dataset to load precomputed embeddings and corresponding IF means
class KronosEmbeddingDataset(Dataset):
    def __init__(self, path, input_marker_indices):
        self.path = path
        self.input_indices = input_marker_indices
        self.file = None  # will be opened on demand

        # Open once to get dataset length and number of markers
        with h5py.File(self.path, 'r') as f:
            self.length = f["embeddings"].shape[0]
            self.num_markers = f["all_if_means"].shape[1]

        self.target_indices = [i for i in range(self.num_markers) if i not in input_marker_indices]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file handle if not already open (one per worker)
        if self.file is None:
            self.file = h5py.File(self.path, 'r')

        embeddings = torch.from_numpy(self.file["embeddings"][idx]).float()
        all_if_means = torch.from_numpy(self.file["all_if_means"][idx]).float()

        input_emb = embeddings[self.input_indices]  # (num_input_markers, 384)
        selected_embeddings_flat = input_emb.flatten()  # (num_input_markers * 384,)
        input_mean = all_if_means[self.input_indices]

        x = torch.cat([selected_embeddings_flat, input_mean], dim=0)
        y = all_if_means[self.target_indices]

        return x, y


# Main model for intensity prediction
class HE_IntensityPredictor(pl.LightningModule):
    def __init__(
        self,
        latent_dim=1536,
        num_markers=17,
        input_marker_indices=None,
        lr=1e-4,
        weight_decay=0.01,
        use_precomputed_embeddings=False
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_markers = num_markers
        self.input_marker_indices = input_marker_indices or []
        self.use_precomputed_embeddings = use_precomputed_embeddings

        self.num_input_markers = len(self.input_marker_indices)
        self.num_output_markers = self.num_markers - self.num_input_markers

        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_output_markers)
        )

    def forward(self, he_images=None, if_images=None, precomputed_embeddings=None):
        feats = precomputed_embeddings
        return self.regressor(feats)

    def compute_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def compute_metrics(self, pred, target):
        mse = F.mse_loss(pred, target)
        spearman_sum = sum([spearman(pred[i], target[i]) for i in range(pred.size(0))])
        avg_spearman = spearman_sum / pred.size(0)
        return mse, avg_spearman

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# Monkey patch for training/validation steps when using precomputed embeddings
def new_training_step(self, batch, batch_idx):
    x, y = batch
    pred = self(precomputed_embeddings=x)
    loss = self.compute_loss(pred, y)
    mse, spearman_corr = self.compute_metrics(pred, y)
    self.log('train_loss', loss, prog_bar=True)
    self.log('train_mse', mse)
    self.log('train_spearman', spearman_corr)
    return loss

def new_validation_step(self, batch, batch_idx):
    x, y = batch
    pred = self(precomputed_embeddings=x)
    loss = self.compute_loss(pred, y)
    mse, spearman_corr = self.compute_metrics(pred, y)
    self.log('val_loss', loss, prog_bar=True)
    self.log('val_mse', mse)
    self.log('val_spearman', spearman_corr)
    return loss


# Setup
input_indices = [6, 11, 13]

# dataset = KronosEmbeddingDataset(
#     "/home/groups/ChangLab/govindsa/KRONOS/scripts/results/train-finetuning_kronos_embeddings_and_means.h5",
#     input_indices
# )

# intensities_pth = '/home/groups/ChangLab/govindsa/KRONOS/batch_test/train_finetuning/predicted_vs_gt_intensities_train_finetuning.csv'
# metrics_pth = '/home/groups/ChangLab/govindsa/KRONOS/batch_test/train_finetuning/metrics_train_finetuning.csv'
dataset = KronosEmbeddingDataset(
    "/home/groups/ChangLab/govindsa/KRONOS/code/results/train_finetuning_embeddings_and_means/train-finetuning_kronos_embeddings_and_means.h5",
    input_indices
)

# intensities_pth = '/home/groups/ChangLab/govindsa/KRONOS/code/results/validation_embeddings_and_means/CRC07/predicted_vs_gt_intensities_train_finetuning.csv'
# metrics_pth = '/home/groups/ChangLab/govindsa/KRONOS/batch_test/train_finetuning/metrics_train_finetuning.csv'

# Index split (80/20)
indices = list(range(len(dataset)))
split = int(0.8 * len(dataset))

train_indices = indices[:split]
val_indices = indices[split:]

from torch.utils.data import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler, num_workers=4, pin_memory=True)

print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

# Get latent dimension from a sample
sample_x, _ = dataset[0]
latent_dim = sample_x.shape[0]

model = HE_IntensityPredictor(
    latent_dim=latent_dim,
    num_markers=17,
    input_marker_indices=input_indices,
    use_precomputed_embeddings=True
)

# Patch training and validation steps
model.training_step = types.MethodType(new_training_step, model)
model.validation_step = types.MethodType(new_validation_step, model)

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/groups/ChangLab/govindsa/KRONOS/code/results/train_finetuning_embeddings_and_means/',                     # ← Update this path
    filename='model-{epoch:02d}-{step}-{val_loss:.4f}',  # Optional: informative filename
    monitor='val_loss',                            # ← Metric to monitor
    mode='min',                                     # 'min' because lower val_loss is better
    save_top_k=3,                                   # ← Save only top 3 checkpoints
    every_n_train_steps=10000,
    save_last=True
)


trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=30,
    precision=16,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback]  # ← Add this line
)


trainer.fit(model, train_loader, val_loader)

# Evaluate on full dataset (batch size 512)
full_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

model.eval()
model.to("cuda")

all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in full_loader:
        x = x.cuda(non_blocking=True)
        preds = model(precomputed_embeddings=x).cpu()
        all_preds.append(preds)
        all_targets.append(y)

# pred_array = torch.cat(all_preds).numpy()
# gt_array = torch.cat(all_targets).numpy()

# # Save predictions vs ground truth
# out_df = pd.DataFrame(gt_array, columns=[f"GT_Marker{i}" for i in range(gt_array.shape[1])])
# for i in range(pred_array.shape[1]):
#     out_df[f"Pred_Marker{i}"] = pred_array[:, i]

# out_df.to_csv(intensities_pth, index=False)
# print(f"Saved: {intensities_pth}")

# # Compute metrics (Spearman, Pearson)
# spearman_corrs = []
# pearson_corrs = []

# for i in range(pred_array.shape[1]):
#     gt = gt_array[:, i]
#     pred = pred_array[:, i]

#     s_r, _ = spearmanr(gt, pred)
#     p_r, _ = pearsonr(gt, pred)

#     spearman_corrs.append(s_r)
#     pearson_corrs.append(p_r)

# marker_names = [f"Marker{i}" for i in range(pred_array.shape[1])]
# metrics_df = pd.DataFrame({
#     "Marker": marker_names,
#     "Spearman": spearman_corrs,
#     "Pearson": pearson_corrs
# })

# metrics_df.to_csv(metrics_pth, index=False)
# print(f"Saved: {metrics_pth}")
