import os
import glob
import math
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, accuracy_score
# ----------------------------
# CONFIG
# ----------------------------
FEATURES_DIR = "/Users/bindusamba/Documents/protein_ligand/features_final"
LABELS_DIR   = "/Users/bindusamba/Documents/protein_ligand/labels"

OUT_DIR = "./cv_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_SPLITS = 10

# Training knobs (keep small/fast; you can increase epochs later)
EPOCHS = 10
BATCH_SIZE = 2
LR = 3e-4
WEIGHT_DECAY = 1e-5

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If your concat features are stored under a subfolder, list it here first.
# Script will try these patterns in order.
FEATURE_PATTERNS = [
    os.path.join(FEATURES_DIR, "concat", "*.npy"),
    os.path.join(FEATURES_DIR, "*concat*.npy"),
    os.path.join(FEATURES_DIR, "*.npy"),
    os.path.join(FEATURES_DIR, "**", "*.npy"),
]

LABEL_PATTERNS = [
    os.path.join(LABELS_DIR, "*.npy"),
    os.path.join(LABELS_DIR, "**", "*.npy"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# DATA DISCOVERY / LOADING
# ----------------------------
def _index_by_stem(paths: List[str]) -> Dict[str, str]:
    """Map filename stem -> full path. Stem is filename without extension."""
    out = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        out[stem] = p
    return out


def find_feature_label_pairs() -> List[Tuple[str, str, str]]:
    """
    Returns list of (protein_id, feature_path, label_path).
    Assumes feature and label share the same filename stem (protein_id).
    """
    # Collect feature files
    feat_files = []
    for pat in FEATURE_PATTERNS:
        feat_files.extend(glob.glob(pat, recursive=True))
    feat_files = sorted(list(set(feat_files)))

    # Collect label files
    lab_files = []
    for pat in LABEL_PATTERNS:
        lab_files.extend(glob.glob(pat, recursive=True))
    lab_files = sorted(list(set(lab_files)))

    if len(feat_files) == 0:
        raise FileNotFoundError(f"No feature .npy files found under: {FEATURES_DIR}")
    if len(lab_files) == 0:
        raise FileNotFoundError(f"No label .npy files found under: {LABELS_DIR}")

    feat_map = _index_by_stem(feat_files)
    lab_map  = _index_by_stem(lab_files)

    common = sorted(set(feat_map.keys()) & set(lab_map.keys()))
    if len(common) == 0:
        raise RuntimeError(
            "No matching feature/label pairs found.\n"
            "Make sure the .npy filenames match (same protein_id stem) in both folders."
        )

    pairs = [(pid, feat_map[pid], lab_map[pid]) for pid in common]
    return pairs


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr


# ----------------------------
# DATASET + COLLATE (variable length)
# ----------------------------
class ProteinResidueDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str, str]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pid, fpath, lpath = self.items[idx]
        X = load_npy(fpath)  # expected shape: (L, C)
        y = load_npy(lpath)  # expected shape: (L,) or (L,1)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if X.ndim != 2:
            raise ValueError(f"{pid}: feature array must be 2D (L,C). Got shape={X.shape}")
        if y.ndim != 1:
            raise ValueError(f"{pid}: label array must be 1D (L,). Got shape={y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{pid}: length mismatch X(L={X.shape[0]}) vs y(L={y.shape[0]})")

        # Convert to float32 / int64
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        return pid, X, y


def pad_collate(batch):
    """
    Pads sequences in batch to max L.
    Returns:
      X: (B, C, Lmax)
      y: (B, Lmax)
      mask: (B, Lmax) 1 where valid residue else 0
      pids: list[str]
    """
    pids, Xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in Xs]
    Lmax = max(lengths)
    C = Xs[0].shape[1]

    X_pad = np.zeros((len(batch), Lmax, C), dtype=np.float32)
    y_pad = np.zeros((len(batch), Lmax), dtype=np.int64)
    mask  = np.zeros((len(batch), Lmax), dtype=np.float32)

    for i, (X, y) in enumerate(zip(Xs, ys)):
        L = X.shape[0]
        X_pad[i, :L, :] = X
        y_pad[i, :L] = y
        mask[i, :L] = 1.0

    # PyTorch expects channels-first for Conv1d: (B, C, L)
    X_t = torch.from_numpy(X_pad).transpose(1, 2)  # (B,C,L)
    y_t = torch.from_numpy(y_pad)                  # (B,L)
    m_t = torch.from_numpy(mask)                   # (B,L)

    return list(pids), X_t, y_t, m_t


# ----------------------------
# MODEL (lightweight 1D CNN)
# ----------------------------
class LiteCNN(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1)  # logits per residue
        )

    def forward(self, x):
        # x: (B,C,L) -> logits: (B,L)
        logits = self.net(x).squeeze(1)
        return logits


def compute_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    # pos_weight = (#neg / #pos) for BCEWithLogitsLoss
    pos = 0
    neg = 0
    for _, _, y, mask in train_loader:
        y = y.numpy()
        m = mask.numpy().astype(bool)
        yy = y[m]
        pos += int((yy == 1).sum())
        neg += int((yy == 0).sum())
    if pos == 0:
        # avoid crash; extremely unlikely
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor):
    """
    logits: (B,L)
    y:      (B,L) 0/1
    mask:   (B,L) 0/1
    """
    y_f = y.float()
    # BCE per element
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight.to(logits.device))
    loss = loss_fn(logits, y_f)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def eval_loader(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_true  = []

    with torch.no_grad():
        for _, X, y, mask in loader:
            X = X.to(DEVICE)
            logits = model(X)  # (B,L)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_np  = y.numpy()
            m_np  = mask.numpy().astype(bool)

            # Flatten valid residues only
            all_probs.append(probs[m_np])
            all_true.append(y_np[m_np])

    probs = np.concatenate(all_probs, axis=0)
    true  = np.concatenate(all_true, axis=0)

    # Safety checks
    if len(np.unique(true)) < 2:
        # AUROC undefined if only one class appears
        auc = float("nan")
        aupr = float("nan")
    else:
        auc  = roc_auc_score(true, probs)
        aupr = average_precision_score(true, probs)

    # Threshold at 0.5 for reporting (you can also tune later)
    pred = (probs >= 0.5).astype(int)
    f1   = f1_score(true, pred, zero_division=0)
    acc  = accuracy_score(true, pred)

    return {"AUROC": auc, "AUPR": aupr, "F1": f1, "ACC": acc}


def roc_points(model: nn.Module, loader: DataLoader):
    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        for _, X, y, mask in loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_np = y.numpy()
            m_np = mask.numpy().astype(bool)
            all_probs.append(probs[m_np])
            all_true.append(y_np[m_np])
    probs = np.concatenate(all_probs, axis=0)
    true  = np.concatenate(all_true, axis=0)
    fpr, tpr, _ = roc_curve(true, probs)
    return fpr, tpr


# ----------------------------
# MAIN 10-FOLD CV
# ----------------------------
def main():
    set_seed(SEED)

    pairs = find_feature_label_pairs()
    protein_ids = [p[0] for p in pairs]
    print(f"[INFO] Found {len(pairs)} protein feature/label pairs.")

    # Determine in_ch from first file
    _, f0, _ = pairs[0]
    X0 = load_npy(f0)
    if X0.ndim != 2:
        raise ValueError(f"First feature file must be (L,C). Got {X0.shape}")
    in_ch = X0.shape[1]
    print(f"[INFO] Detected feature channels in_ch = {in_ch}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(protein_ids), start=1):
        train_items = [pairs[i] for i in train_idx]
        val_items   = [pairs[i] for i in val_idx]

        train_ds = ProteinResidueDataset(train_items)
        val_ds   = ProteinResidueDataset(val_items)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

        pos_weight = compute_pos_weight(train_loader).to(DEVICE)

        model = LiteCNN(in_ch=in_ch, hidden=128, dropout=0.2).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_auc = -1.0
        best_state = None

        for ep in range(1, EPOCHS + 1):
            model.train()
            running = 0.0
            steps = 0

            for _, X, y, mask in train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                mask = mask.to(DEVICE)

                opt.zero_grad(set_to_none=True)
                logits = model(X)
                loss = masked_bce_loss(logits, y, mask, pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                running += loss.item()
                steps += 1

            # quick validation AUROC to keep best model
            metrics = eval_loader(model, val_loader)
            val_auc = metrics["AUROC"]
            avg_loss = running / max(steps, 1)

            print(f"[Fold {fold:02d}] Ep{ep:02d} loss={avg_loss:.4f}  valAUROC={val_auc:.4f}  valAUPR={metrics['AUPR']:.4f}  valF1={metrics['F1']:.4f}")

            if not math.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # final fold metrics
        fold_m = eval_loader(model, val_loader)
        fold_m["fold"] = fold
        fold_metrics.append(fold_m)

        # ROC points for mean ROC
        fpr, tpr = roc_points(model, val_loader)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        # save fold metrics
        with open(os.path.join(OUT_DIR, f"fold_{fold:02d}_metrics.json"), "w") as f:
            json.dump(fold_m, f, indent=2)

    # summarize
    def mean_std(key: str):
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary = {}
    for k in ["AUROC", "AUPR", "F1", "ACC"]:
        mu, sd = mean_std(k)
        summary[k] = {"mean": mu, "std": sd}

    with open(os.path.join(OUT_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 10-FOLD CV SUMMARY (mean ± std) ===")
    for k, v in summary.items():
        print(f"{k:5s}: {v['mean']:.4f} ± {v['std']:.4f}")

    # mean ROC
    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    mean_tpr[-1] = 1.0
    roc_out = np.vstack([mean_fpr, mean_tpr]).T
    np.savetxt(os.path.join(OUT_DIR, "mean_roc_cv.csv"), roc_out, delimiter=",", header="fpr,tpr", comments="")

    print(f"\n[Saved] {os.path.join(OUT_DIR, 'mean_roc_cv.csv')}")
    print(f"[Saved] {os.path.join(OUT_DIR, 'cv_summary.json')}")
    print("[Done]")


if __name__ == "__main__":
    main()
