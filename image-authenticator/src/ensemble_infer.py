import os, sys

# ---- Fix import path so 'backend' can be found ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ---------------------------------------------------

import argparse, yaml, torch
from torch.utils.data import DataLoader
from utils.dataset import build_datasets
from backend.models.factory import build_model
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm   # ✅ Progress bar

# --- Helper function to log metrics automatically ---
def log_results_to_csv(model_name, accuracy, precision, recall, f1, csv_path="results_log.csv"):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Timestamp"])
        writer.writerow([
            model_name,
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="list of ckpts")
    ap.add_argument("--models", nargs="+", required=True, help="parallel list of model names")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    assert len(args.weights) == len(args.models)

    cfg = yaml.safe_load(open(args.config))
    _, val_ds, test_ds = build_datasets(cfg["data_root"], cfg["img_size"])
    ds = val_ds if args.split == "val" else test_ds

    dl = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []
    for name, w in zip(args.models, args.weights):
        m = build_model(name, 2).to(device)
        state = torch.load(w, map_location=device)
        if "model" in state:  # if you saved {"model": state_dict}
            state = state["model"]
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    correct, n = 0, 0
    predictions = []
    y_true, y_pred = [], []
    all_ai_probs = []  # ✅ For ROC curve

    all_paths = [s[0] for s in ds.samples]
    idx = 0

    print(f"\nRunning ensemble inference on {args.split} split...\n")

    # ✅ Progress bar for batches
    for x, y in tqdm(dl, desc="Evaluating", unit="batch"):
        x, y = x.to(device), y.to(device)
        probs = None
        per_model_preds = []

        for m in models:
            logits = m(x)
            p = F.softmax(logits, dim=1)
            per_model_preds.append(p.argmax(1).cpu())
            probs = p if probs is None else probs + p

        probs /= len(models)
        pred = probs.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)

        y_true += y.cpu().tolist()
        y_pred += pred.cpu().tolist()
        all_ai_probs.extend(probs[:, 1].cpu().numpy())  # ✅ store AI probs

        batch_paths = all_paths[idx: idx + x.size(0)]
        idx += x.size(0)

        for path, true, ens in zip(batch_paths, y.cpu(), pred.cpu()):
            row = {"image": path, "true": true.item(), "ensemble_pred": ens.item()}
            for i, pm in enumerate(per_model_preds):
                row[f"pred_{args.models[i]}"] = pm[i].item()
            predictions.append(row)

    # --- Compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nEnsemble: {' + '.join(args.models)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=ds.classes))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # --- Save predictions ---
    os.makedirs("checkpoints", exist_ok=True)
    out_path = os.path.join("checkpoints", f"ensemble_{args.split}_predictions.csv")
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")

    # --- Log metrics automatically ---
    log_results_to_csv(f"Ensemble ({' + '.join(args.models)})", acc, prec, rec, f1)

    # === Generate and Save Plots ===
    os.makedirs("results", exist_ok=True)

    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ds.classes)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - Ensemble ({' + '.join(args.models)})")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "confusion_matrix.png"), dpi=300)
    plt.close()
    print("✅ Saved: results/confusion_matrix.png")

    # --- ROC Curve Plot ---
    if len(all_ai_probs) == len(y_true):  # sanity check
        fpr, tpr, _ = roc_curve(y_true, all_ai_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Ensemble Model')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join("results", "roc_curve.png"), dpi=300)
        plt.close()
        print("✅ Saved: results/roc_curve.png")

if __name__ == "__main__":
    main()
