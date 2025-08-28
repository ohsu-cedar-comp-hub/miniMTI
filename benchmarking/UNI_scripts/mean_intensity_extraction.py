import torch
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.utils.data import Dataset
import h5py

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
        embeddings = torch.from_numpy(self.embeddings[idx]).float()
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

        # CRITICAL FIX: When input_indices=[], input_means is empty tensor
        # So actual input dim = latent_dim + 0 = latent_dim
        input_dim = latent_dim + self.num_input_markers

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024),  # This should be latent_dim when input_indices=[]
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


# Load checkpoint
checkpoint_path = "/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/mean_intensities_train_finetuning/model-epoch=07-step=450000_mints.ckpt"
intensities_pth = '/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/validation_embeddings_and_means/CRC07_10_13_mean_intensities/mints_scores/predicted_vs_gt_intensities_crc710131.csv'
metrics_pth = '/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/validation_embeddings_and_means/CRC07_10_13_mean_intensities/mints_scores/metrics_train1.csv'

# Setup - using empty input_indices as in training
input_indices = []
dataset = EmbeddingDataset("/home/groups/ChangLab/govindsa/Panel_Reduction_Project/UNI/results/validation_embeddings_and_means/CRC07_10_13_mean_intensities/UNI_CRC07_10_13_embeddings_and_means.h5", input_indices)

# Get dimensions from first sample
sample_x, sample_y = dataset[0]
latent_dim = sample_x.shape[0]  # Should be pure embedding dim since input_indices=[]

print(f"Input dimension: {latent_dim}")
print(f"Output dimension: {sample_y.shape[0]}")
print(f"Input marker indices: {input_indices}")

model = HE_IntensityPredictor(
    latent_dim=latent_dim,
    num_markers=17,
    input_marker_indices=input_indices,
    use_precomputed_embeddings=True
)

model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
model = model.to("cuda")
model.eval()

# Prepare DataLoader for evaluation
full_loader = DataLoader(dataset, batch_size=64, shuffle=False)

all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in full_loader:
        x = x.cuda()
        preds = model(precomputed_embeddings=x).cpu()
        all_preds.append(preds)
        all_targets.append(y)

# Stack predictions and targets
pred_array = torch.cat(all_preds).numpy()
gt_array = torch.cat(all_targets).numpy()

# Since input_indices=[], all 17 markers are targets
target_indices = list(range(17))

# Save predicted vs ground truth CSV
out_df = pd.DataFrame(gt_array, columns=[f"GT_Marker{i}" for i in target_indices])
for i in range(pred_array.shape[1]):
    out_df[f"Pred_Marker{i}"] = pred_array[:, i]

out_df.to_csv(intensities_pth, index=False)
print(f"Saved predicted vs GT intensities to: {intensities_pth}")

# Calculate metrics per marker
spearman_corrs = []
pearson_corrs = []

for i in range(pred_array.shape[1]):
    gt = gt_array[:, i]
    pred = pred_array[:, i]

    s_r, _ = spearmanr(gt, pred)
    p_r, _ = pearsonr(gt, pred)

    spearman_corrs.append(s_r)
    pearson_corrs.append(p_r)

# Save metrics to CSV
marker_names = [f"Marker{i}" for i in range(pred_array.shape[1])]
metrics_df = pd.DataFrame({
    "Marker": marker_names,
    "Spearman": spearman_corrs,
    "Pearson": pearson_corrs
})

metrics_df.to_csv(metrics_pth, index=False)
print(f"Saved marker prediction metrics to: {metrics_pth}")

print(f"\nEvaluation Summary:")
print(f"Number of samples: {len(pred_array)}")
print(f"Number of markers predicted: {pred_array.shape[1]}")
print(f"Mean Spearman correlation: {np.mean(spearman_corrs):.4f}")
print(f"Mean Pearson correlation: {np.mean(pearson_corrs):.4f}")