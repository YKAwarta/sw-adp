"""
SW-ADP Stage 4: Shared data loading and utilities.
Loads Track B tensors and prepares PyTorch datasets for autoencoder training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SegmentDataset(Dataset):
    """PyTorch dataset for OPSSAT-AD segments with mask support."""

    def __init__(self, X, mask, y=None):
        """
        Args:
            X: np.ndarray [N, seq_len, n_features]
            mask: np.ndarray [N, seq_len] (1=real, 0=pad)
            y: np.ndarray [N] optional labels
        """
        self.X = torch.FloatTensor(X)
        self.mask = torch.FloatTensor(mask)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.mask[idx], self.y[idx]
        return self.X[idx], self.mask[idx]


def masked_mse_loss(reconstructed, original, mask):
    """
    MSE loss computed only on non-padded timesteps.

    Args:
        reconstructed: [batch, seq_len, n_features]
        original: [batch, seq_len, n_features]
        mask: [batch, seq_len] with 1=real, 0=pad
    Returns:
        scalar loss (mean over batch)
    """
    # Expand mask to match feature dimension
    mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
    sq_error = (reconstructed - original) ** 2  # [batch, seq_len, n_features]
    masked_error = sq_error * mask_expanded  # zero out padded positions

    # Sum per sample, divide by (real_timesteps * n_features)
    real_counts = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # [batch, 1, 1]
    n_features = original.shape[-1]
    per_sample_mse = masked_error.sum(dim=(1, 2)) / (real_counts.squeeze() * n_features + 1e-8)
    return per_sample_mse.mean()


def per_segment_masked_mse(reconstructed, original, mask):
    """
    Per-segment reconstruction error (for anomaly scoring).

    Args:
        reconstructed: [N, seq_len, n_features]
        original: [N, seq_len, n_features]
        mask: [N, seq_len]
    Returns:
        errors: [N] per-segment MSE
    """
    mask_expanded = mask.unsqueeze(-1)
    sq_error = (reconstructed - original) ** 2
    masked_error = sq_error * mask_expanded
    real_counts = mask.sum(dim=1)
    n_features = original.shape[-1]
    per_sample = masked_error.sum(dim=(1, 2)) / (real_counts * n_features + 1e-8)
    return per_sample


def load_data(tensor_path, track_a_sw_path=None):
    """
    Load Track B tensors and return split data.

    Returns dict with:
        X_telem, X_telem_sw: full arrays
        y, mask, train_mask: full arrays
        For convenience, also provides filtered views.
    """
    data = np.load(tensor_path, allow_pickle=True)
    result = {
        'X_telem': data['X_telem'],         # [2098, 128, 1]
        'X_telem_sw': data['X_telem_sw'],    # [2098, 128, 7]
        'y': data['y'],                       # [2098]
        'mask': data['mask'],                 # [2098, 128]
        'train_mask': data['train_mask'].astype(bool),  # [2098]
    }

    # Load SW summary stats for MP baseline if path provided
    if track_a_sw_path is not None:
        import pandas as pd
        df = pd.read_csv(track_a_sw_path)
        sw_cols = [c for c in df.columns if c.startswith('sw_')]
        result['sw_summary'] = df[sw_cols].values

    return result


def build_ae_dataloaders(X, mask, y, train_mask, batch_size=32, val_frac=0.2, seed=42):
    """
    Build train/val/test dataloaders for autoencoder training.

    Training uses NORMAL segments only (y==0 & train_mask==True).
    Validation uses a held-out fraction of normal training segments.
    Test uses all test segments (train_mask==False).

    Returns:
        train_loader, val_loader, test_loader, test_y
    """
    rng = np.random.RandomState(seed)

    # Normal training segments
    normal_train_idx = np.where((y == 0) & train_mask)[0]
    rng.shuffle(normal_train_idx)

    n_val = int(len(normal_train_idx) * val_frac)
    val_idx = normal_train_idx[:n_val]
    train_idx = normal_train_idx[n_val:]

    # Test: ALL test segments (normal + anomalous)
    test_idx = np.where(~train_mask)[0]

    train_ds = SegmentDataset(X[train_idx], mask[train_idx])
    val_ds = SegmentDataset(X[val_idx], mask[val_idx])
    test_ds = SegmentDataset(X[test_idx], mask[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train (normal only): {len(train_idx)} segments")
    print(f"  Val (normal only):   {len(val_idx)} segments")
    print(f"  Test (all):          {len(test_idx)} segments ({(y[test_idx]==1).sum()} anomalous)")

    return train_loader, val_loader, test_loader, y[test_idx]
