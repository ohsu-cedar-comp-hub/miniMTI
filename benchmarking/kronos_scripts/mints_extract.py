import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import h5py
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import spearman_corrcoef as spearman
import pytorch_lightning as pl

# Dataset class (same as before)
class KronosEmbeddingDataset(Dataset):
    def __init__(self, path, input_marker_indices):
        self.path = path
        self.input_indices = input_marker_indices
        self.file = None

        with h5py.File(self.path, 'r') as f:
            self.length = f["embeddings"].shape[0]
            self.num_markers = f["all_if_means"].shape[1]

        self.target_indices = [i for i in range(self.num_markers) if i not in input_marker_indices]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')

        embeddings = torch.from_numpy(self.file["embeddings"][idx]).float()
        all_if_means = torch.from_numpy(self.file["all_if_means"][idx]).float()

        input_emb = embeddings[self.input_indices]
        selected_embeddings_flat = input_emb.flatten()
        input_mean = all_if_means[self.input_indices]

        x = torch.cat([selected_embeddings_flat, input_mean], dim=0)
        y = all_if_means[self.target_indices]

        return x, y

# Model class (same as before)
class HE_IntensityPredictor(pl.LightningModule):
    def __init__(
        self,
        latent_dim=1536,
        num_markers=17,
        input_marker_indices=None,
        lr=1e-4,
        weight_decay=0.01,
        use_precomputed_embeddings=True
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

# --- CONFIG ---
input_indices = [6, 11, 13]
checkpoint_path = '/home/groups/ChangLab/govindsa/KRONOS/code/results/train_finetuning_embeddings_and_means/model-epoch=04-step=510000-val_loss=265-mints.ckpt'  # ← Replace this
h5_path = '/home/groups/ChangLab/govindsa/KRONOS/code/results/validation_embeddings_and_means/KRONOS_CRC07_10_13_embeddings_and_means.h5'  # ← Replace this
intensities_pth = '/home/groups/ChangLab/govindsa/KRONOS/code/results/train_finetuning_embeddings_and_means/mint_scores/predicted_vs_gt_intensities_final.csv'  # ← Replace this
metrics_pth = '/home/groups/ChangLab/govindsa/KRONOS/code/results/train_finetuning_embeddings_and_means/mint_scores/metrics_final.csv'  # ← Replace this

# --- LOAD DATASET ---
dataset = KronosEmbeddingDataset(h5_path, input_indices)
full_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# --- INFER LATENT DIM ---
sample_x, _ = dataset[0]
latent_dim = sample_x.shape[0]

# --- LOAD MODEL FROM CHECKPOINT ---
model = HE_IntensityPredictor(
    latent_dim=latent_dim,
    num_markers=17,
    input_marker_indices=input_indices,
    use_precomputed_embeddings=True
)
model = model.load_from_checkpoint(
    checkpoint_path,
    latent_dim=latent_dim,
    num_markers=17,
    input_marker_indices=input_indices,
    use_precomputed_embeddings=True
)

model.eval()
model.to("cuda")

# --- INFERENCE ---
all_preds, all_targets = [], []
with torch.no_grad():
    for x, y in full_loader:
        x = x.cuda(non_blocking=True)
        preds = model(precomputed_embeddings=x).cpu()
        all_preds.append(preds)
        all_targets.append(y)

pred_array = torch.cat(all_preds).numpy()
gt_array = torch.cat(all_targets).numpy()

# --- SAVE PREDICTIONS ---
out_df = pd.DataFrame(gt_array, columns=[f"GT_Marker{i}" for i in range(gt_array.shape[1])])
for i in range(pred_array.shape[1]):
    out_df[f"Pred_Marker{i}"] = pred_array[:, i]

out_df.to_csv(intensities_pth, index=False)
print(f"Saved predictions: {intensities_pth}")

# --- COMPUTE METRICS ---
spearman_corrs = []
pearson_corrs = []

for i in range(pred_array.shape[1]):
    gt = gt_array[:, i]
    pred = pred_array[:, i]

    s_r, _ = spearmanr(gt, pred)
    p_r, _ = pearsonr(gt, pred)

    spearman_corrs.append(s_r)
    pearson_corrs.append(p_r)

marker_names = [f"Marker{i}" for i in range(pred_array.shape[1])]
metrics_df = pd.DataFrame({
    "Marker": marker_names,
    "Spearman": spearman_corrs,
    "Pearson": pearson_corrs
})

metrics_df.to_csv(metrics_pth, index=False)
print(f"Saved metrics: {metrics_pth}")
