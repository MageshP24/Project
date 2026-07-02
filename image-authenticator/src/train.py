import argparse, os, yaml, torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils.dataset import build_datasets
from backend.models.factory import build_model

def train_one_epoch(model, loader, opt, loss_fn, device, scaler=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds, gts = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        n += x.size(0)
        preds.extend(p.cpu().tolist()); gts.extend(y.cpu().tolist())
    f1 = f1_score(gts, preds, average="macro")
    return total_loss / n, correct / n, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="effnet_b3", choices=["effnet_b3","vit_b16"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, _ = build_datasets(cfg["data_root"], cfg["img_size"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(args.model, num_classes=2).to(device)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)))

    best_f1, patience, es_pat = -1, 0, cfg["early_stopping_patience"]
    os.makedirs(cfg["model_dir"], exist_ok=True)

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, loss_fn, device, scaler)
        va_loss, va_acc, va_f1 = evaluate(model, val_dl, loss_fn, device)
        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1, patience = va_f1, 0
            ckpt = os.path.join(cfg["model_dir"], f"{args.model}_best.pt")
            torch.save({"model": model.state_dict()}, ckpt)
            print(f"Saved {ckpt}")
        else:
            patience += 1
            if patience >= es_pat:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()
