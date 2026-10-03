from __future__ import annotations
import matplotlib.pyplot as plt

def plot_run(result: dict, path: str) -> None:
    history = result["history"]
    cfg = result["cfg"]
    epochs = history["epoch"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].axvline(result["summary"]["best_epoch"], color='r', linestyle='--', label='Best Epoch')
    axes[0].set_title(f"Loss - {cfg['exp_id']}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    
    # 2. Accuracy & Macro-F1
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    if "val_macro_f1" in history and history["val_macro_f1"]:
        axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1")
    axes[1].axvline(result["summary"]["best_epoch"], color='r', linestyle='--')
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    
    # 3. Grad Norm
    axes[2].plot(epochs, history["grad_norm"], label="Grad Norm")
    axes[2].set_title("Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for res in results:
        history = res["history"]
        exp_id = res["cfg"]["exp_id"]
        epochs = history["epoch"]
        if metric in history:
            ax.plot(epochs, history[metric], label=exp_id)
            
    ax.set_title(title if title else f"Compare {metric}")
    ax.set_xlabel("Epoch")
    ax.legend()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
