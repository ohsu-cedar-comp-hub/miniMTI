import h5py
import torch
import gc
from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file_path = h5_file
        # Open file in worker processes, not main process
        self.h5_file = None
        self.tokens = None
        self._initialize_dataset()

    def _initialize_dataset(self):
        # Initialize once when actually needed
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')
            self.tokens = self.h5_file['tokens']

    def __len__(self):
        self._initialize_dataset()
        return len(self.tokens)

    def __getitem__(self, idx):
        self._initialize_dataset()
        # Use astype instead of torch.tensor for better memory management
        tokens = torch.from_numpy(self.tokens[idx].astype('int64')).long()
        return tokens
    
    def __del__(self):
        # Cleanup resources when the dataset is destroyed
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.tokens = None
            gc.collect()

def get_tokenized_dataloader(train_file, batch_size, num_workers=4):
    
    train_data = TokenizedDataset(train_file)
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),  # Only use persistent workers if num_workers > 0
        pin_memory=True,
        drop_last=True  # Avoid partial batches which could cause memory variations
    )
    
    # Return the dataloaders and the metadata
    return train_loader