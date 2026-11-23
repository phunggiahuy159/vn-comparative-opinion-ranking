"""Generate LaTeX tables and number macros for the reports from results/ (so numbers are never copied by hand).

Writes report/tables/{absa,synthetic,real}.tex (complete tabular environments) and report/tables/numbers.tex
(\newcommand macros used in the text). Run from anywhere:  python report/make_tables.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"


def f(x, d=3):
    """0.xxx with a leading zero; minus sign in math mode."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{abs(x):.{d}f}"
    return f"$-${s}" if x < 0 else s


def load(p):
    p = RES / p
    return json.loads(p.read_text()) if p.exists() else None


def absa():
    base, ph = load("absa_baseline_summary.json"), load("absa_phobert-base-v2/metrics.json")
    asp = ["BATTERY", "CAMERA", "PERFORMANCE", "SCREEN"]
    rows = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
            r"UIT-ViSFD test, micro-F1 & Detection & Aspect\#polarity & Battery & Camera & Perf. & Screen \\", r"\midrule"]
    names = {"multinomial_nb": "TF-IDF + multinomial NB", "linear_svm": "TF-IDF + linear SVM",
             "logreg": "TF-IDF + logistic regression"}
    for k, name in names.items():
        m = (base or {}).get("models", {}).get(k)
        if m:
            rows.append(f"{name} & {f(m['aspect_detection_microF1'])} & {f(m['aspect_polarity_microF1'])} & -- & -- & -- & -- \\\\")
    if ph:
        per = " & ".join(f(ph["per_aspect"][a]["pair_microF1"]) for a in asp)
        rows.append(rf"PhoBERT-base-v2, multi-task & \textbf{{{f(ph['aspect_detection_microF1'])}}} & "
                    rf"\textbf{{{f(ph['aspect_polarity_microF1'])}}} & {per} \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def synthetic():
    """E1 table: mean ± sd over seeds; in each column the best mean (any row, ablations included) is bold."""
    e1 = load("synthetic/e1_main.json")["spearman"]
    G = ["0.0", "0.5", "0.8"]
    main = [("price only", "Price only"), ("sentiment only", "Sentiment only"), ("PageRank", "PageRank"), ("HITS", "HITS"),
            ("pRank [Zhang 2010]", "pRank"), ("Bradley-Terry", "Bradley--Terry (BT)"), ("BT + sentiment", "BT + sentiment"),
            ("EDR (ours)", "EDR (ours)")]
    abl = [("EDR: γ = 0", r"\quad $-$ $\gamma$ fixed at 0"), ("EDR: γ learned jointly", r"\quad $-$ $\gamma$ learned jointly"),
           ("EDR: no hedonic prior", r"\quad $-$ no hedonic prior")]
    keys = [k for k, _ in main + abl if k in e1[G[0]]]
    best = {g: max(e1[g][k][0] for k in keys) for g in G}

    def cell(k, g):
        m, sd = e1[g][k]
        txt = f(m)
        txt = rf"\textbf{{{txt}}}" if abs(m - best[g]) < 5e-4 else txt
        return txt + rf"{{\scriptsize$\pm${f(sd, 2)}}}"

    rows = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Ranker & $\gamma{=}0$ & $\gamma{=}0.5$ & $\gamma{=}0.8$ \\", r"\midrule"]
    rows += [f"{lab} & " + " & ".join(cell(k, g) for g in G) + r" \\" for k, lab in main if k in e1[G[0]]]
    rows.append(r"\midrule")
    rows += [f"{lab} & " + " & ".join(cell(k, g) for g in G) + r" \\" for k, lab in abl if k in e1[G[0]]]
    rows.append(r"\quad selected $\gamma$ & " + " & ".join(f(e1[g]["chosen γ"][0], 2) + rf"{{\scriptsize$\pm${f(e1[g]['chosen γ'][1], 2)}}}" for g in G) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def real():
    ev = load("real_eval.json") or {}
    proxy_name = {"BATTERY": "Battery & mAh", "SCREEN": "Screen & peak nits", "CAMERA": "Camera & main MP, weak",
                  "PERFORMANCE": "Performance & AnTuTu"}
    cols = [("price only", "Price only"), ("sentiment", "Sentiment"), ("pRank", "pRank"), ("EDR", "EDR")]
    rows = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", "Aspect & Proxy ($n$) & " + " & ".join(c for _, c in cols) + r" \\",
            r"\midrule"]
    for a in ["BATTERY", "SCREEN", "CAMERA", "PERFORMANCE"]:
        p = (ev.get(a) or {}).get("proxy")
        if not isinstance(p, dict) or "n" not in p:
            continue
        cells = []
        for k, _ in cols:
            x = p.get(k) or {}
            lo, hi = (x.get("ci95") or [None, None])
            cells.append(f"{f(x.get('spearman'), 2)} {{\\scriptsize[{f(lo, 2)}, {f(hi, 2)}]}}")
        asp, prox = proxy_name[a].split(" & ")
        rows.append(f"{asp} & {prox} ({p['n']}) & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def numbers():
    """Macros for numbers quoted in the text."""
    m = {}
    st = load("real_pipeline_stats.json") or {}
    src = st.get("reviews_by_source", {})
    m["NReviews"] = f"{st.get('reviews', 0):,}".replace(",", "{,}")
    m["NReviewsTGDD"] = f"{src.get('tgdd', 0):,}".replace(",", "{,}")
    m["NReviewsCPS"] = f"{src.get('cellphones', 0):,}".replace(",", "{,}")
    m["NProducts"] = str(st.get("products", 0))
    m["NLinkedComps"] = str(st.get("linked_comparisons", 0))
    e2 = load("synthetic/e2_breakeven.json")
    if e2:
        edr, price = e2["series"]["EDR (ours)"][0], e2["series"]["price only"][0]
        m["MarginLow"] = f(edr[0] - price[0], 2)
        m["MarginHigh"] = f(edr[-1] - price[-1], 2)
    e1 = load("synthetic/e1_main.json")
    if e1:
        sp = e1["spearman"]
        m["pRankGzero"] = f(sp["0.0"]["pRank [Zhang 2010]"][0], 2)
        m["pRankGhigh"] = f(sp["0.8"]["pRank [Zhang 2010]"][0], 2)
        m["ECE"] = f(e1.get("ece_gamma0.8"), 2)
    ev = load("real_eval.json") or {}
    for a, key in [("BATTERY", "RealPriceBattery"), ("CAMERA", "RealPriceCamera")]:
        m[key] = f(((ev.get(a) or {}).get("proxy") or {}).get("price only", {}).get("spearman"), 2)
    m["RealSentScreen"] = f(((ev.get("SCREEN") or {}).get("proxy") or {}).get("sentiment", {}).get("spearman"), 2)
    for a, key in [("BATTERY", "RealPriceBatteryCI"), ("CAMERA", "RealPriceCameraCI")]:
        lo, hi = ((ev.get(a) or {}).get("proxy") or {}).get("price only", {}).get("ci95", [None, None])
        m[key] = f"[{f(lo, 2)}, {f(hi, 2)}]"
    ek = st.get("edge_kinds", {})
    m["NEdgesCatCat"] = str(ek.get("cat-cat", 0))
    m["NEdgesOld"] = str(ek.get("cat-ext", 0) + ek.get("ext-cat", 0))
    m["NEdgesBrand"] = str(sum(v for k, v in ek.items() if "group" in k))
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items())


def main():
    OUT.mkdir(exist_ok=True)
    for name, fn in [("absa", absa), ("synthetic", synthetic), ("real", real), ("numbers", numbers)]:
        (OUT / f"{name}.tex").write_text(f"% generated by report/make_tables.py from results/ — do not edit\n{fn()}\n")
        print("wrote", OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
