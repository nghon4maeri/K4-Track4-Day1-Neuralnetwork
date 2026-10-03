import os
import sys
import json
import subprocess
import torch
import numpy as np

# Thêm thư mục code vào sys.path để import
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from data import prepare_data
from train import DEFAULT_CFG, run_experiment, final_eval
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx

OUT_DIR = "."
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUT_DIR}/results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

print("Loading data...")
data = prepare_data(device, val_fraction=0.2, seed=42, processed_dir="data/processed")

# Danh sách các cấu hình sẽ chạy (đều tăng lên 40 epochs)
configs = []

# 1. Baseline (SGD) - 2 seeds
for s in [1, 2]:
    configs.append({
        **DEFAULT_CFG,
        "exp_id": f"base-s{s}",
        "group": "baseline",
        "description": f"Baseline SGD s{s}",
        "optimizer": "sgd_momentum",
        "lr": 0.1,
        "epochs": 40,
        "seed": s
    })

# 2. Adam
configs.append({
    **DEFAULT_CFG,
    "exp_id": "opt-adam-lr1e-3",
    "group": "optimizer",
    "description": "Adam optimizer",
    "optimizer": "adam",
    "lr": 1e-3,
    "epochs": 40
})

# 3. AdamW with Weight Decay
configs.append({
    **DEFAULT_CFG,
    "exp_id": "opt-adamw-wd0.01",
    "group": "optimizer",
    "description": "AdamW with weight_decay=0.01",
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.01,
    "epochs": 40
})

# 4. M-wide Architecture
configs.append({
    **DEFAULT_CFG,
    "exp_id": "arch-m-wide",
    "group": "architecture",
    "description": "M-wide (512, 256)",
    "optimizer": "adam",
    "lr": 1e-3,
    "hidden": (512, 256),
    "epochs": 40
})

# 5. M-deep Architecture
configs.append({
    **DEFAULT_CFG,
    "exp_id": "arch-m-deep",
    "group": "architecture",
    "description": "M-deep (256, 128, 64)",
    "optimizer": "adam",
    "lr": 1e-3,
    "hidden": (256, 128, 64),
    "epochs": 40
})

# 6. Dropout
configs.append({
    **DEFAULT_CFG,
    "exp_id": "reg-dropout-0.3",
    "group": "regularization",
    "description": "Dropout p=0.3",
    "optimizer": "adam",
    "lr": 1e-3,
    "dropout": 0.3,
    "epochs": 40
})

# 7. Gradient Clipping
configs.append({
    **DEFAULT_CFG,
    "exp_id": "reg-clip-1.0",
    "group": "regularization",
    "description": "Gradient Clipping max_norm=1.0",
    "optimizer": "adam",
    "lr": 1e-3,
    "clip_norm": 1.0,
    "epochs": 40
})

results_list = []
best_macro_f1 = 0
best_cfg = None
best_res = None

# Chạy tất cả các cấu hình
for cfg in configs:
    print(f"\n--- Running: {cfg['exp_id']} ({cfg['description']}) ---")
    res = run_experiment(cfg, data)
    save_result(res, f"{OUT_DIR}/results")
    plot_run(res, f"{OUT_DIR}/figures/{cfg['exp_id']}.png")
    results_list.append(res)
    
    val_f1 = res['summary']['val_macro_f1']
    print(f"-> Val Macro-F1: {val_f1:.4f}")
    
    # Tìm cấu hình tốt nhất (loại baseline để luôn chọn model mới nộp)
    if cfg["group"] != "baseline" and val_f1 > best_macro_f1:
        best_macro_f1 = val_f1
        best_cfg = cfg
        best_res = res

# Vẽ biểu đồ so sánh cho optimizer
opt_results = [r for r in results_list if r["cfg"]["group"] in ["baseline", "optimizer"]]
plot_compare(opt_results, "val_loss", f"{OUT_DIR}/figures/compare_optimizer_val_loss.png", title="Val Loss: Optimizers")
plot_compare(opt_results, "val_macro_f1", f"{OUT_DIR}/figures/compare_optimizer_macro_f1.png", title="Val Macro-F1: Optimizers")

# Vẽ biểu đồ so sánh kiến trúc
arch_results = [r for r in results_list if r["cfg"]["group"] == "architecture" or r["cfg"]["exp_id"] == "opt-adam-lr1e-3"]
plot_compare(arch_results, "val_macro_f1", f"{OUT_DIR}/figures/compare_architecture_macro_f1.png", title="Val Macro-F1: Architectures")

# Đánh giá cấu hình tốt nhất trên tập Eval
print(f"\n--- Best Config found: {best_cfg['exp_id']} (Val F1: {best_macro_f1:.4f}) ---")
final_eval(best_cfg, best_res, data, f"{OUT_DIR}/predictions_eval.csv")

# Chấm điểm bằng evaluate.py
print("Evaluating predictions...")
eval_cmd = [sys.executable, f"scripts/evaluate.py", "--pred", f"{OUT_DIR}/predictions_eval.csv", "--out", f"{OUT_DIR}/eval_result.json"]
# Dùng dict env để set UTF-8
my_env = os.environ.copy()
my_env["PYTHONIOENCODING"] = "utf-8"
result = subprocess.run(eval_cmd, capture_output=True, text=True, env=my_env)
print(result.stdout)
if result.stderr:
    print("Error:", result.stderr)

# Đọc kết quả eval để điền vào Excel
eval_data = None
if os.path.exists(f"{OUT_DIR}/eval_result.json"):
    with open(f"{OUT_DIR}/eval_result.json", "r") as f:
        eval_data = json.load(f)

# Viết ra Excel
rows = []
for r in results_list:
    if r["cfg"]["exp_id"] == best_cfg["exp_id"]:
        rows.append(to_row(r, eval_scores=eval_data))
    else:
        rows.append(to_row(r))

write_xlsx(rows, f"templates/experiment_table_template.xlsx", f"{OUT_DIR}/experiments.xlsx")
print(f"Đã cập nhật xong experiments.xlsx")
