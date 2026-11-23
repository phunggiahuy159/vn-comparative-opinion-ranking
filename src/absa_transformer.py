"""Multi-task ABSA on UIT-ViSFD with a pretrained Vietnamese encoder (PhoBERT-base-v2 by default).

Architecture: shared encoder -> [CLS] + mean-pooled tokens -> one 4-way head per aspect
(none / Positive / Negative / Neutral). Joint detection + polarity, as in course L06-02.
Loss: per-aspect cross-entropy with class weights (handles "none" dominance), averaged over aspects.
Selects the best epoch on dev (aspect#polarity micro-F1), reports test, saves predictions + metrics.

Runs on Apple MPS, CUDA (Colab/Kaggle) or CPU:
  .venv/bin/python src/absa_transformer.py --model vinai/phobert-base-v2 --epochs 6
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ASPECTS = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL", "PERFORMANCE",
           "PRICE", "SCREEN", "SER&ACC", "STORAGE", "OTHERS"]
POLS = ["none", "Positive", "Negative", "Neutral"]
ROOT = Path(__file__).resolve().parent.parent


def parse(label):
    y = [0] * len(ASPECTS)
    for tok in re.findall(r"\{([^}]*)\}", label):
        a, p = (tok.split("#") + ["Positive"])[:2]  # OTHERS has no polarity -> treat as present
        if a in ASPECTS:
            y[ASPECTS.index(a)] = POLS.index(p)
    return y


def preprocess(text, segment):
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if segment:
        from pyvi import ViTokenizer
        t = ViTokenizer.tokenize(t)
    return t


class ABSAModel(nn.Module):
    def __init__(self, name, n_aspects, n_pols, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.Linear(2 * h, n_aspects * n_pols)
        self.na, self.np = n_aspects, n_pols

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = out[:, 0]
        m = attention_mask.unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        z = self.drop(torch.cat([cls, mean], -1))
        return self.heads(z).view(-1, self.na, self.np)


def metrics(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    det_t, det_p = (yt > 0).astype(int), (yp > 0).astype(int)
    pair_t = np.concatenate([(yt == k).astype(int) for k in range(1, len(POLS))], 1)
    pair_p = np.concatenate([(yp == k).astype(int) for k in range(1, len(POLS))], 1)
    per = {}
    for i, a in enumerate(ASPECTS):
        per[a] = {"det_F1": f1_score(det_t[:, i], det_p[:, i], zero_division=0),
                  "pair_microF1": f1_score(np.stack([(yt[:, i] == k) for k in (1, 2, 3)], 1).astype(int),
                                           np.stack([(yp[:, i] == k) for k in (1, 2, 3)], 1).astype(int),
                                           average="micro", zero_division=0)}
    return {"aspect_detection_microF1": f1_score(det_t, det_p, average="micro"),
            "aspect_detection_macroF1": f1_score(det_t, det_p, average="macro", zero_division=0),
            "aspect_polarity_microF1": f1_score(pair_t, pair_p, average="micro"),
            "per_aspect": per}


def batches(enc, ys, bs, shuffle):
    idx = list(range(len(ys)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield ({k: torch.tensor([enc[k][j] for j in b]) for k in ("input_ids", "attention_mask")},
               torch.tensor([ys[j] for j in b]))


def pad(enc, max_len, pad_id):
    ids = [x[:max_len] + [pad_id] * (max_len - len(x[:max_len])) for x in enc["input_ids"]]
    att = [[1] * min(len(x), max_len) + [0] * (max_len - min(len(x), max_len)) for x in enc["input_ids"]]
    return {"input_ids": ids, "attention_mask": att}


@torch.no_grad()
def predict(model, enc, ys, bs, device):
    model.eval()
    preds, probs = [], []
    for x, _ in batches(enc, ys, bs, False):
        logits = model(**{k: v.to(device) for k, v in x.items()}).float()
        p = logits.softmax(-1).cpu()
        probs.append(p)
        preds.append(p.argmax(-1))
    return torch.cat(preds).tolist(), torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/phobert-base-v2")
    ap.add_argument("--segment", type=int, default=1, help="pyvi word segmentation (needed for PhoBERT)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--amp", type=int, default=1, help="fp16 autocast on CUDA")
    ap.add_argument("--keep_ckpt", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tag = args.tag or f"{args.model.split('/')[-1]}_s{args.seed}"
    out_dir = ROOT / "results" / f"absa_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset("visolex/ViSFD")["train"].to_pandas()
    df["text"] = [preprocess(t, args.segment) for t in df.comment]
    df["y"] = df.label.map(parse)
    split = {s: df[df.type == s].reset_index(drop=True) for s in ("train", "dev", "test")}

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = {s: pad(tok(list(d.text), add_special_tokens=True, truncation=True, max_length=args.max_len),
                  args.max_len, tok.pad_token_id) for s, d in split.items()}
    ys = {s: list(d.y) for s, d in split.items()}

    model = ABSAModel(args.model, len(ASPECTS), len(POLS)).to(device)
    # class weights per aspect: inverse sqrt frequency (smoothed)
    Ytr = np.array(ys["train"])
    w = np.stack([1 / np.sqrt(np.bincount(Ytr[:, i], minlength=len(POLS)) + 10) for i in range(len(ASPECTS))])
    w = w / w.mean(1, keepdims=True)
    cw = torch.tensor(w, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": args.lr},
                             {"params": model.heads.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(ys["train"]) / args.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    use_amp = bool(args.amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, log = -1, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0
        for step, (x, y) in enumerate(batches(enc["train"], ys["train"], args.bs, True)):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**{k: v.to(device) for k, v in x.items()})
            y = y.to(device)
            logits = logits.float()
            loss = sum(nn.functional.cross_entropy(logits[:, i], y[:, i], weight=cw[i])
                       for i in range(len(ASPECTS))) / len(ASPECTS)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
            if step % 50 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f}", flush=True)
        dev_pred, _ = predict(model, enc["dev"], ys["dev"], 64, device)
        m = metrics(ys["dev"], dev_pred)
        rec = {"epoch": ep, "train_loss": tot / (step + 1), "dev_det": m["aspect_detection_microF1"],
               "dev_pair": m["aspect_polarity_microF1"], "sec": round(time.time() - t0)}
        log.append(rec); print(rec, flush=True)
        if rec["dev_pair"] > best:
            best = rec["dev_pair"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_pred, test_prob = predict(model, enc["test"], ys["test"], 64, device)
    m = metrics(ys["test"], test_pred)
    m.update({"model": args.model, "segment": bool(args.segment), "device": device, "args": vars(args), "log": log})
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    np.save(out_dir / "test_probs.npy", test_prob)
    if not args.keep_ckpt:
        (out_dir / "best.pt").unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in m.items() if k not in ("per_aspect", "log", "args")}, indent=2))
    for a, v in m["per_aspect"].items():
        print(f"{a:12s} det_F1={v['det_F1']:.3f}  pair_microF1={v['pair_microF1']:.3f}")


if __name__ == "__main__":
    main()
