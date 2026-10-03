from __future__ import annotations
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)

def load_split(processed_dir: str = "data/processed"):
    train_data = np.load(f"{processed_dir}/train.npz")
    eval_data = np.load(f"{processed_dir}/eval.npz")
    
    X_train_full, y_train_full = train_data["X"], train_data["y"]
    X_eval, y_eval, eval_row_id = eval_data["X"], eval_data["y"], eval_data["row_id"]
    
    assert X_train_full.shape[1] == 54 and X_train_full.dtype == np.float32
    assert y_train_full.dtype == np.int64
    
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id

def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val

def fit_standardizer(X_tr):
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    std[std == 0] = 1.0 
    return mean, std

def apply_standardizer(X, mean, std):
    X_copy = X.copy()
    X_copy[:, :N_NUMERIC] = (X_copy[:, :N_NUMERIC] - mean) / std
    return X_copy

def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)
    
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)
    
    data = {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id
    }
    
    print(f"Kích thước X_tr: {X_tr.shape[0]}, X_val: {X_val.shape[0]}, X_eval: {X_eval.shape[0]}")
    majority_class = np.bincount(y_tr).argmax()
    majority_acc = (y_val == majority_class).mean()
    print(f"Accuracy chiến lược đoán lớp đa số trên val: {majority_acc:.4f}")
    
    return data

def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    N = X.shape[0]
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)
        
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield X[idx], y[idx]
