from __future__ import annotations
import time
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=0.05,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,
    precision="fp32",
    seed=1,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    
    den = precision + recall
    f1 = np.divide(2 * precision * recall, den, out=np.zeros_like(tp, dtype=float), where=den > 0)
    return float(f1.mean())

@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    preds = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds)

@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    model.eval()
    total_loss = 0.0
    preds = []
    
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model(xb)
        
        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        else:
            y_onehot = F.one_hot(yb, num_classes=logits.size(-1)).float()
            loss = F.mse_loss(logits, y_onehot, reduction="sum")
            
        total_loss += loss.item()
        preds.append(logits.argmax(dim=1))
        
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item()
    
    preds_np = preds.cpu().numpy()
    y_np = y.cpu().numpy()
    cm = confusion_matrix(y_np, preds_np, labels=range(logits.size(-1)))
    macro_f1 = macro_f1_from_confusion(cm)
    
    return {
        "loss": total_loss / X.shape[0],
        "acc": acc,
        "macro_f1": macro_f1
    }

def compute_loss(logits, y, loss_name: str):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.size(-1)).float()
        return F.mse_loss(logits, y_onehot)
    raise ValueError("Unknown loss")

def run_experiment(cfg: dict, data: dict) -> dict:
    set_seed(cfg["seed"])
    
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(X_tr.device)
    
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"], 
        weight_decay=cfg["weight_decay"], momentum=cfg["momentum"]
    )
    
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" else None
    
    history = {k: [] for k in ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s"]}
    
    step0_metrics = evaluate(model, X_val, y_val, cfg["loss"])
    step0_loss = step0_metrics["loss"]
    
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False
    
    sub_size = min(50000, X_tr.shape[0])
    sub_idx = torch.randperm(X_tr.shape[0])[:sub_size]
    X_tr_sub, y_tr_sub = X_tr[sub_idx], y_tr[sub_idx]
    
    generator = torch.Generator(device=X_tr.device)
    generator.manual_seed(cfg["seed"])
    
    for epoch in range(1, cfg["epochs"] + 1):
        if diverged: break
        
        start_time = time.time()
        model.train()
        epoch_grad_norm = 0.0
        steps = 0
        
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
            optimizer.zero_grad(set_to_none=True)
            
            if cfg["precision"] == "fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
                scaler.scale(loss).backward()
                if cfg["clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            elif cfg["precision"] == "bf16":
                with torch.autocast(device_type=X_tr.device.type, dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                optimizer.step()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                optimizer.step()
                
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break
                
            epoch_grad_norm += gn
            steps += 1
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        
        train_metrics = evaluate(model, X_tr_sub, y_tr_sub, cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"])
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(epoch_grad_norm / steps if steps > 0 else 0)
        history["epoch_time_s"].append(epoch_time)
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_mem = 0.0
        
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if not diverged else float('nan'),
        "final_val_loss": history["val_loss"][-1] if not diverged else float('nan'),
        "val_acc": history["val_acc"][best_epoch-1] if best_epoch > 0 else 0,
        "val_macro_f1": history["val_macro_f1"][best_epoch-1] if best_epoch > 0 else 0,
        "time_per_epoch_s": np.mean(history["epoch_time_s"]) if len(history["epoch_time_s"]) > 0 else 0,
        "peak_mem_MB": peak_mem,
        "diverged": diverged
    }
    
    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }

def write_predictions(row_id, preds, path: str) -> None:
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)

def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"Predictions saved to {pred_path}")
