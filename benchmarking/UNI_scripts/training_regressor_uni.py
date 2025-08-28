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
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from pytorch_lightning.callbacks import ModelCheckpoint
# Dataset to load precomputed embeddings and corresponding IF means
class EmbeddingDataset(Dataset):
    def __init__(self, path, input_marker_indices):
        self.h5_file = h5py.File(path, 'r')
        self.embeddings = self.h5_file["embeddings"]
        self.all_if_means = self.h5_file["all_if_means"]
        self.input_indices = input_marker_indices
        self.target_indices = [i for i in range(17) if i not in input_marker_indices]
        self.input_means = self.all_if_means[:, self.input_indices]
        self.target_means = self.all_if_means[:, self.target_indices]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embeddings=torch.from_numpy(self.embeddings[idx]).float()
        input_means = torch.from_numpy(self.all_if_means[idx, self.input_indices]).float()
        target_means = torch.from_numpy(self.all_if_means[idx, self.target_indices]).float()
        x = torch.cat([embeddings, input_means], dim=0)
        y = target_means
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
# input_indices = [6, 11, 13]
input_indices=[]
dataset = EmbeddingDataset("/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/validation_embeddings_and_means/CRC07_10_13_mean_intensities/UNI_CRC07_10_13_embeddings_and_means.h5", input_indices)
# intensities_pth='/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/mean_intensities_train_finetuning/predicted_vs_gt_intensities_train_finetuning.csv'
# metrics_pth='/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/mean_intensities_train_finetuning/metrics_train_finetuning.csv'
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
sample_x, _ = dataset[0]
latent_dim = sample_x.shape[0]

model = HE_IntensityPredictor(
    latent_dim=latent_dim,
    num_markers=17,
    input_marker_indices=input_indices,
    use_precomputed_embeddings=True
)


import types
model.training_step = types.MethodType(new_training_step, model)
model.validation_step = types.MethodType(new_validation_step, model)

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/validation_embeddings_and_means/CRC07_10_13_mean_intensities_arc/',  # <-- your desired checkpoint directory
    filename='model-{epoch:02d}-{step}',  # optional naming pattern
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


import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# Load 10000 embeddings
full_loader = DataLoader(dataset, batch_size=64, shuffle=False)

model.eval()
model.to("cuda")

all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in full_loader:
        x = x.cuda()
        preds = model(precomputed_embeddings=x).cpu()
        all_preds.append(preds)
        all_targets.append(y)

# Stack all results
pred_array = torch.cat(all_preds).numpy()  # shape: (1000, N)
gt_array   = torch.cat(all_targets).numpy()

# # Save as CSV
# out_df = pd.DataFrame(gt_array, columns=[f"GT_Marker{i}" for i in range(gt_array.shape[1])])
# for i in range(pred_array.shape[1]):
#     out_df[f"Pred_Marker{i}"] = pred_array[:, i]

# out_df.to_csv(intensities_pth, index=False)
# print("Saved: predicted_vs_gt_intensities.csv")

# import pandas as pd
# from scipy.stats import spearmanr, pearsonr
# from skimage.metrics import structural_similarity as ssim
# import numpy as np

# spearman_corrs = []
# pearson_corrs = []
# ssim_scores = []

# for i in range(pred_array.shape[1]):
#     gt = gt_array[:, i]
#     pred = pred_array[:, i]

#     # Spearman correlation
#     s_r, _ = spearmanr(gt, pred)
#     spearman_corrs.append(s_r)

#     # Pearson correlation
#     p_r, _ = pearsonr(gt, pred)
#     pearson_corrs.append(p_r)

# # Save to DataFrame
# marker_names = [f"Marker{i}" for i in range(pred_array.shape[1])]
# metrics_df = pd.DataFrame({
#     "Marker": marker_names,
#     "Spearman": spearman_corrs,
#     "Pearson": pearson_corrs
# })

# # Save to CSV
# metrics_df.to_csv(metrics_pth, index=False)
# print("Saved: marker_prediction_metrics.csv")

