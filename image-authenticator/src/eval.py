import os, sys
# ---- FIX: make sure Python can find the backend folder ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ------------------------------------------------------------

import argparse, yaml, torch
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import build_datasets
from backend.models.factory import build_model
from sklearn.metrics import classification_report, confusion_matrix

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["effnet_b3", "vit_b16"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    _, _, test_ds = build_datasets(cfg["data_root"], cfg["img_size"])
    dl = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, 2).to(device)
    state = torch.load(args.weights, map_location=device)["model"]
    model.load_state_dict(state)
    loss_fn = nn.CrossEntropyLoss()

    y_true, y_pred = [], []
    tot_loss, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)
        y_true += y.cpu().tolist()
        y_pred += logits.argmax(1).cpu().tolist()

    print("Test Loss:", tot_loss / n)
    print(classification_report(y_true, y_pred, target_names=test_ds.classes))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
