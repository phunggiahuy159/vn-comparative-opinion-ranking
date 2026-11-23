"""Apply the trained multi-task PhoBERT ABSA model to new reviews (CPU by default — no GPU needed).

Usage (library):
    from absa_infer import ABSAPredictor
    pred = ABSAPredictor("results/absa_phobert-base-v2")      # dir with best.pt + metrics.json
    out = pred.predict(["pin trâu, camera hơi kém"])           # -> list[dict aspect -> (label, prob)]
"""
import json
from pathlib import Path

import numpy as np
import torch

from absa_transformer import ASPECTS, POLS, ABSAModel, pad, preprocess


class ABSAPredictor:
    def __init__(self, run_dir, device="cpu", bs=64):
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "metrics.json").read_text())
        self.args = meta["args"]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.args["model"])
        self.model = ABSAModel(self.args["model"], len(ASPECTS), len(POLS))
        self.model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        self.model.to(device).eval()
        self.device, self.bs = device, bs

    @torch.no_grad()
    def predict_proba(self, texts):
        texts = [preprocess(t or "", self.args["segment"]) for t in texts]
        probs = []
        for i in range(0, len(texts), self.bs):
            enc = pad(self.tok(texts[i:i + self.bs], truncation=True, max_length=self.args["max_len"]),
                      self.args["max_len"], self.tok.pad_token_id)
            x = {k: torch.tensor(v, device=self.device) for k, v in enc.items()}
            probs.append(self.model(**x).float().softmax(-1).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, len(ASPECTS), len(POLS)))

    def predict(self, texts):
        P = self.predict_proba(texts)
        out = []
        for p in P:
            d = {}
            for i, a in enumerate(ASPECTS):
                k = int(p[i].argmax())
                if k > 0:
                    d[a] = (POLS[k], float(p[i, k]))
            out.append(d)
        return out
