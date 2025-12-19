#!/usr/bin/env python3
# esm_extract.py
# Bulk per-residue ESM2 embeddings -> NumPy arrays, robust to long sequences.

import argparse, os, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import esm

# ----------------------------
# Defaults (tweak as needed)
# ----------------------------
DEFAULT_MODEL = "esm2_t33_650M_UR50D"  # 1280-d; good perf vs. compute
MAX_TOKENS = 1022                     # effective per-seq token limit for residues
WINDOW = 1000                         # sliding window length (<= MAX_TOKENS)
STRIDE = 900                          # overlap for stitching
LAYER = 33                            # final layer for esm2_t33_650M
DTYPE_HALF = True                     # fp16 on GPU to save memory

def read_fasta(fp: Path) -> List[Tuple[str, str]]:
    recs = []
    pid, seq = None, []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            if s.startswith(">"):
                if pid is not None:
                    recs.append((pid, "".join(seq)))
                pid = s[1:].split()[0]
                seq = []
            else:
                seq.append(s)
    if pid is not None:
        recs.append((pid, "".join(seq)))
    return recs

@torch.no_grad()
def embed_sequence_per_res(model, alphabet, device, seq: str, layer: int) -> np.ndarray:
    """Return (L, D) per-residue embeddings; handles long sequences by sliding windows."""
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    def run_chunk(subseq: str) -> torch.Tensor:
        data = [("seq", subseq)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)
        out = model(toks, repr_layers=[layer], return_contacts=False)
        rep = out["representations"][layer]  # (1, 1+L+1, D)
        rep = rep[:, 1:1+len(subseq), :]     # strip BOS/EOS
        return rep.squeeze(0)                # (L, D)

    L = len(seq)
    if L <= MAX_TOKENS:
        rep = run_chunk(seq)                 # (L, D)
        return rep.cpu().float().numpy()

    # Sliding window with overlap
    pieces = []
    positions = []
    i = 0
    while i < L:
        j = min(i + WINDOW, L)
        chunk = seq[i:j]
        rep = run_chunk(chunk)               # (len(chunk), D)
        pieces.append(rep)
        positions.append((i, j))
        if j == L: 
            break
        i += STRIDE

    # Stitch with simple averaging in overlaps
    D = pieces[0].shape[1]
    full = torch.zeros(L, D, device=pieces[0].device)
    wts  = torch.zeros(L, 1, device=pieces[0].device)

    for (i, j), rep in zip(positions, pieces):
        length = j - i
        full[i:j] += rep[:length]
        wts[i:j]  += 1.0

    full = full / wts.clamp_min(1.0)
    return full.cpu().float().numpy()

def main():
    ap = argparse.ArgumentParser(description="Extract per-residue ESM2 embeddings -> NumPy")
    ap.add_argument("-i", "--fasta", required=True, type=Path)
    ap.add_argument("-o", "--outdir", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="e.g., esm2_t33_650M_UR50D")
    ap.add_argument("--layer", type=int, default=LAYER, help="repr layer to export")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model, alphabet = getattr(esm.pretrained, args.model)()
    model = model.to(device)
    if device.type == "cuda" and DTYPE_HALF:
        model = model.half()  # mixed precision ok for inference

    recs = read_fasta(args.fasta)
    print(f"[ESM] Model={args.model}  layer={args.layer}  device={device}  sequences={len(recs)}")

    for pid, seq in recs:
        out = args.outdir / f"{pid}.npy"
        if out.exists() and not args.overwrite:
            continue
        if not seq:
            print(f"[WARN] empty sequence for {pid}; skipping")
            continue
        emb = embed_sequence_per_res(model, alphabet, device, seq, layer=args.layer)  # (L, D)
        np.save(out, emb)
        print(f"[ESM] wrote {out.name} shape={emb.shape}")

if __name__ == "__main__":
    main()
