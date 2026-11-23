"""Which aspects drive Vietnamese buyers' star ratings? (real data, UIT-ViSFD)

Each comment has a 1–5 star rating and aspect#polarity labels. Regress stars on aspect polarity indicators
(+1 positive, −1 negative, 0 absent/neutral) → coefficient = how many stars an aspect moves the overall rating.
Also reports the negativity asymmetry (|effect of a negative| vs effect of a positive) per aspect, with
bootstrap 95% CIs. Used to (a) weight aspects in an overall ranking, (b) interpret buyer priorities.
"""
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
ASPECTS = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL", "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE"]


def main(B=500):
    df = load_dataset("visolex/ViSFD")["train"].to_pandas()
    n = len(df)
    pos = np.zeros((n, len(ASPECTS))); neg = np.zeros((n, len(ASPECTS)))
    for r, lab in enumerate(df.label):
        for tok in re.findall(r"\{([^}]*)\}", lab):
            if "#" in tok:
                a, p = tok.split("#")
                if a in ASPECTS:
                    (pos if p == "Positive" else neg if p == "Negative" else np.zeros_like(pos))[r, ASPECTS.index(a)] = 1
    X = np.hstack([pos, neg]); y = df.n_star.values.astype(float)
    rng = np.random.default_rng(0)
    coefs = []
    for b in range(B + 1):
        idx = np.arange(n) if b == 0 else rng.integers(0, n, n)
        coefs.append(LinearRegression().fit(X[idx], y[idx]).coef_)
    coefs = np.array(coefs); point, lo, hi = coefs[0], np.percentile(coefs[1:], 2.5, 0), np.percentile(coefs[1:], 97.5, 0)
    k = len(ASPECTS)
    rows = []
    for i, a in enumerate(ASPECTS):
        rows.append({"aspect": a, "freq_pos": int(pos[:, i].sum()), "freq_neg": int(neg[:, i].sum()),
                     "stars_if_pos": float(round(point[i], 3)), "ci_pos": [float(round(lo[i], 2)), float(round(hi[i], 2))],
                     "stars_if_neg": float(round(point[k + i], 3)), "ci_neg": [float(round(lo[k + i], 2)), float(round(hi[k + i], 2))],
                     "neg_pos_ratio": float(round(abs(point[k + i]) / max(point[i], 1e-6), 2))})
    rows.sort(key=lambda r: r["stars_if_neg"])
    r2 = LinearRegression().fit(X, y).score(X, y)
    print(f"R² = {r2:.3f}  (n = {n})")
    print(f"{'aspect':12s}{'+ effect':>10s}{'95% CI':>18s}{'− effect':>10s}{'95% CI':>18s}{'|−|/+':>7s}{'#neg':>6s}")
    for r in rows:
        print(f"{r['aspect']:12s}{r['stars_if_pos']:>10.2f}{str(r['ci_pos']):>18s}{r['stars_if_neg']:>10.2f}{str(r['ci_neg']):>18s}{r['neg_pos_ratio']:>7.1f}{r['freq_neg']:>6d}")
    (ROOT / "results" / "aspect_importance.json").write_text(json.dumps({"r2": r2, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
