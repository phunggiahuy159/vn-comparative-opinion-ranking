"""Baseline ABSA on UIT-ViSFD (Vietnamese smartphone reviews).

Two subtasks (as in Phan et al., 2021):
  1. Aspect category detection   — multi-label over 10 aspects (+OTHERS)
  2. Aspect-level polarity        — for each aspect: none / Positive / Negative / Neutral
Features: TF-IDF (word 1-2gram + char 2-5gram). Classifiers (one per aspect): logistic regression, linear SVM and
multinomial Naive Bayes, the classical sentiment classifiers. Also reports the share of reviews with comparative cues.

Usage: .venv/bin/python src/absa_baseline.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_recall_fscore_support

ASPECTS = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL", "PERFORMANCE",
           "PRICE", "SCREEN", "SER&ACC", "STORAGE", "OTHERS"]
POLS = ["none", "Positive", "Negative", "Neutral"]
COMP_CUE = re.compile(r"(?:hơn|kém hơn|không bằng|bằng|so với|nhất|thua|ăn đứt|vượt)", re.I)
OUT = Path(__file__).resolve().parent.parent / "results"
MODELS = {"logreg": lambda: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
          "linear_svm": lambda: LinearSVC(C=0.5, class_weight="balanced"),
          "multinomial_nb": lambda: MultinomialNB(alpha=0.05, fit_prior=False)}


def parse(label: str) -> dict:
    y = {a: "none" for a in ASPECTS}
    for tok in re.findall(r"\{([^}]*)\}", label):
        if "#" in tok:
            a, p = tok.split("#")
            y[a] = p
        elif tok in y:
            y[tok] = "Positive"  # OTHERS carries no polarity; treat as present
    return y


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def main():
    df = load_dataset("visolex/ViSFD")["train"].to_pandas()
    df["text"] = df.comment.map(norm)
    Y = pd.DataFrame([parse(l) for l in df.label])
    tr, te = (df.type == "train").values, (df.type == "test").values

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    X_tr = hstack([word.fit_transform(df.text[tr]), char.fit_transform(df.text[tr])]).tocsr()
    X_te = hstack([word.transform(df.text[te]), char.transform(df.text[te])]).tocsr()

    summary = {"split_sizes": df.type.value_counts().to_dict(),
               "comments_with_comparative_cue": float(df.text.str.contains(COMP_CUE).mean()), "models": {}}
    OUT.mkdir(exist_ok=True)
    for name, make in MODELS.items():
        rows, det_true, det_pred, pair_true, pair_pred = [], [], [], [], []
        for a in ASPECTS:
            y_tr, y_te = Y[a][tr].values, Y[a][te].values
            clf = make().fit(X_tr, y_tr)
            p = clf.predict(X_te)
            dt, dp = (y_te != "none").astype(int), (p != "none").astype(int)
            det_true.append(dt); det_pred.append(dp)
            # aspect#polarity pairs: correct only if aspect present and polarity matches
            for pol in POLS[1:]:
                pair_true.append((y_te == pol).astype(int)); pair_pred.append((p == pol).astype(int))
            P, R, F, _ = precision_recall_fscore_support(dt, dp, average="binary", zero_division=0)
            mask = (y_te != "none") & (p != "none")
            pol_acc = float((y_te[mask] == p[mask]).mean()) if mask.any() else float("nan")
            rows.append({"aspect": a, "support_test": int(dt.sum()), "det_P": P, "det_R": R, "det_F1": F,
                         "polarity_acc_on_detected": pol_acc})
        det_true, det_pred = np.stack(det_true, 1), np.stack(det_pred, 1)
        pair_true, pair_pred = np.stack(pair_true, 1), np.stack(pair_pred, 1)
        summary["models"][name] = {
            "aspect_detection_microF1": f1_score(det_true, det_pred, average="micro"),
            "aspect_detection_macroF1": f1_score(det_true, det_pred, average="macro"),
            "aspect_polarity_microF1": f1_score(pair_true, pair_pred, average="micro"),
            "aspect_polarity_macroF1": f1_score(pair_true, pair_pred, average="macro", zero_division=0)}
        pd.DataFrame(rows).round(3).to_csv(OUT / f"absa_baseline_per_aspect_{name}.csv", index=False)
        print(name, json.dumps(summary["models"][name]), flush=True)
    (OUT / "absa_baseline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
