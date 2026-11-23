"""Synthetic experiments for EDR — one simulated world, three knobs, all baselines.

World (per seed): 40 phones, 6 brands, 2 aspects (BATTERY evaluated). Ground truth follows a hedonic market:
  q = μ_p + c·z(log price) + brand effect + noise           (true quality = what a benchmark would measure)
  ê = c·z + brand effect                                    (what buyers expect from price & brand)
  sentiment counts:  P(+) = σ(α (q − γ ê))                  γ = how price-relative reviewers are
  comparisons:       Bradley–Terry(q), Zipf popularity, 30% brand-only mentions, direction flipped w.p. ε
Knobs: γ (price-relativity), ε (extraction error), m (comparisons per aspect).

E1  main table: γ ∈ {0, .5, .8} — all rankers + EDR ablations + chosen γ + calibration   (20 seeds)
E2  break-even: ε sweep — when does comparison mining beat ranking by price?                (10 seeds)
E3  sparsity:   m sweep                                                                     (10 seeds)
Outputs: results/synthetic/*.json, figures/*.pdf|png
Usage: .venv/bin/python experiments/synthetic.py [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import bradley_terry, build_matrix, hits, pagerank, prank, sentiment_prior, zfuse  # noqa: E402
from edr import EDR, select_gamma  # noqa: E402

AS = ["BATTERY", "CAMERA"]
OUT = ROOT / "results" / "synthetic"
FIG = ROOT / "figures"


def world(seed, gamma=0.8, m=300, eps=0.1, brand_frac=0.3, n=40, nb=6, alpha=1.5):
    rng = np.random.default_rng(seed)
    prods = [f"p{i:02d}" for i in range(n)]
    brand = rng.integers(0, nb, n)
    logp = rng.normal(size=n); z = (logp - logp.mean()) / logp.std()
    price = dict(zip(prods, np.exp(16 + 0.5 * logp)))
    bq = rng.normal(scale=0.4, size=(nb, len(AS)))
    c = np.array([0.7, 0.9])
    ehat = z[:, None] * c[None, :] + bq[brand]
    q = rng.normal(scale=0.6, size=n)[:, None] + ehat + rng.normal(scale=0.5, size=(n, len(AS)))
    pop = rng.dirichlet(np.ones(n) * 0.5)
    groups = {f"b{k}": [prods[i] for i in range(n) if brand[i] == k] for k in range(nb)}
    comps = []
    for ai, a in enumerate(AS):
        for _ in range(m):
            i, j = rng.choice(n, 2, replace=False, p=pop)
            w, l = (i, j) if rng.random() < 1 / (1 + np.exp(-(q[i, ai] - q[j, ai]))) else (j, i)
            W, L = prods[w], prods[l]
            brand_level = rng.random() < brand_frac and brand[i] != brand[j]
            if brand_level:                                   # "hơn samsung": only the competitor's brand is named
                if w == j: W = f"group:b{brand[j]}"
                else: L = f"group:b{brand[j]}"
            if rng.random() < eps:                            # extraction error flips the direction
                W, L = L, W
            comps.append({"aspect": a, "winner": W, "loser": L, "brand_level": brand_level, "eps": eps})
    senti = {}
    for ai, a in enumerate(AS):
        for i in range(n):
            k = rng.poisson(25 * pop[i] * n)
            pos = rng.binomial(k, 1 / (1 + np.exp(-alpha * (q[i, ai] - gamma * ehat[i, ai]))))
            senti[(prods[i], a)] = {"pos": int(pos), "neg": int(k - pos)}
    return dict(prods=prods, q=q, price=price, brand={prods[i]: f"b{brand[i]}" for i in range(n)},
                comps=comps, senti=senti, groups=groups)


def evaluate_world(W, ablations=False, calibration=False):
    P, a = W["prods"], "BATTERY"
    truth = W["q"][:, 0]
    rho = lambda s: float(spearmanr([s[p] for p in P], truth).statistic)
    ent = [c for c in W["comps"] if not c["brand_level"]]          # baselines cannot use brand-only mentions
    nodes, M = build_matrix(ent, a, P)
    senti_score = dict(zip(nodes, sentiment_prior(nodes, a, W["senti"])))
    bt = bradley_terry(nodes, M)
    r = {"price only": rho(W["price"]), "sentiment only": rho(senti_score), "PageRank": rho(pagerank(nodes, M)), "HITS": rho(hits(nodes, M)),
         "pRank [Zhang 2010]": rho(prank(nodes, M, a, W["senti"])), "Bradley-Terry": rho(bt),
         "BT + sentiment": rho(zfuse(bt, senti_score))}
    fitkw = dict(price=W["price"], brand=W["brand"], groups=W["groups"])
    m, g, _ = select_gamma(P, AS, W["comps"], W["senti"], **fitkw)
    r["EDR (ours)"] = rho(m.quality(a)); r["_gamma"] = g
    if ablations:
        for name, opts in {"EDR: γ = 0": {"price_relative": False}, "EDR: γ learned jointly": {},
                           "EDR: no hedonic prior": {"hedonic_prior": False, "gamma": g}}.items():
            r[name] = rho(EDR(**opts).fit(P, AS, W["comps"], W["senti"], **fitkw).quality(a))
    if calibration:
        pr, lab = [], []
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                pr.append(m.prob_better(P[i], P[j], a)); lab.append(float(truth[i] > truth[j]))
        r["_calib"] = (pr, lab)
    return r


def agg(runs, key):
    v = np.array([x[key] for x in runs], dtype=float)
    return float(v.mean()), float(v.std())


# ---------------------------------------------------------------------------------------------- plotting
# Colour per method (validated categorical palette) plus distinct markers/line styles, so it still reads in greyscale.
STY = {"EDR (ours)":         dict(color="#2a78d6", ls="-",  marker="o", lw=1.7, mfc="#2a78d6"),
       "BT + sentiment":     dict(color="#1baf7a", ls="--", marker="s", lw=1.3, mfc="white"),
       "pRank [Zhang 2010]": dict(color="#eb6834", ls=":",  marker="^", lw=1.5, mfc="white"),
       "price only":         dict(color="#8a8984", ls="-.", marker=None, lw=1.1, mfc="white")}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "font.size": 9,
                         "mathtext.fontset": "stix", "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight"})
    return plt


def line_figure(x, series, xlabel, fname, ylim=(0.2, 0.9)):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for name, (mu, sd) in series.items():
        mu, sd = np.array(mu), np.array(sd); st = STY[name]
        ax.plot(x, mu, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=4, mfc=st["mfc"],
                mec=st["color"], mew=0.9, label=name, zorder=3)
        if st["marker"]:
            ax.fill_between(x, mu - sd / np.sqrt(10), mu + sd / np.sqrt(10), color=st["color"], alpha=0.12, lw=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(r"Spearman $\rho$ vs. true quality"); ax.set_ylim(*ylim)
    ax.legend(fontsize=7.5, loc="lower left" if "ε" in xlabel or "varepsilon" in xlabel else "lower right", handlelength=2.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=220)
    plt.close(fig)


def calibration_bins(pr, lab):
    pr, lab = np.array(pr), np.array(lab)
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(pr, bins) - 1, 0, 9)
    keep = [k for k in range(10) if (idx == k).sum() > 20]
    return [float(pr[idx == k].mean()) for k in keep], [float(lab[idx == k].mean()) for k in keep]


def calibration_figure(xs, ys, fname):
    plt = _style()
    fig, ax = plt.subplots(figsize=(2.3, 2.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", lw=0.9, label="perfect")
    ax.plot(xs, ys, color="#2a78d6", lw=1.6, marker="o", ms=3.5, label="EDR")
    ax.set_xlabel(r"predicted $P(A \succ B)$"); ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7.5, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=220)
    plt.close(fig)


def redraw():
    """Redraw all figures from saved JSON (no simulation)."""
    e2 = json.loads((OUT / "e2_breakeven.json").read_text()); e3 = json.loads((OUT / "e3_sparsity.json").read_text())
    line_figure(e2["eps"], e2["series"], "comparison-extraction error rate ε", "breakeven")
    line_figure(e3["m"], e3["series"], "comparisons per aspect", "sparsity")
    e1 = json.loads((OUT / "e1_main.json").read_text())
    if "calibration_bins" in e1:
        calibration_figure(*e1["calibration_bins"], "calibration")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots", action="store_true", help="only redraw figures from results/synthetic/*.json")
    args = ap.parse_args()
    if args.plots:
        return redraw()
    S1, S2 = (4, 3) if args.quick else (20, 10)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)

    # E1 -------------------------------------------------------------------------------------------
    e1, calib = {}, ([], [])
    for gamma in (0.0, 0.5, 0.8):
        runs = [evaluate_world(world(s, gamma=gamma), ablations=True, calibration=(gamma == 0.8)) for s in range(S1)]
        keys = [k for k in runs[0] if not k.startswith("_")]
        e1[gamma] = {k: agg(runs, k) for k in keys}
        e1[gamma]["chosen γ"] = agg(runs, "_gamma")
        if gamma == 0.8:
            for r in runs:
                calib[0].extend(r["_calib"][0]); calib[1].extend(r["_calib"][1])
    pr, lab = np.array(calib[0]), np.array(calib[1])
    bins = np.clip((pr * 10).astype(int), 0, 9)
    ece = float(sum(abs(pr[bins == k].mean() - lab[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any()))
    cb = calibration_bins(pr, lab)
    (OUT / "e1_main.json").write_text(json.dumps({"spearman": {str(k): v for k, v in e1.items()}, "ece_gamma0.8": ece,
                                                  "calibration_bins": cb}, indent=1))
    calibration_figure(*cb, "calibration")

    # E2 -------------------------------------------------------------------------------------------
    eps_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    e2 = {e: [evaluate_world(world(s, gamma=0.8, eps=e)) for s in range(S2)] for e in eps_grid}
    show = ["EDR (ours)", "BT + sentiment", "pRank [Zhang 2010]", "price only"]
    ser2 = {k: ([agg(e2[e], k)[0] for e in eps_grid], [agg(e2[e], k)[1] for e in eps_grid]) for k in show}
    (OUT / "e2_breakeven.json").write_text(json.dumps({"eps": eps_grid, "series": ser2}, indent=1))
    line_figure(eps_grid, ser2, "comparison-extraction error rate ε", "breakeven")

    # E3 -------------------------------------------------------------------------------------------
    m_grid = [50, 100, 200, 400]
    e3 = {m: [evaluate_world(world(s, gamma=0.8, m=m)) for s in range(S2)] for m in m_grid}
    ser3 = {k: ([agg(e3[m], k)[0] for m in m_grid], [agg(e3[m], k)[1] for m in m_grid]) for k in show}
    (OUT / "e3_sparsity.json").write_text(json.dumps({"m": m_grid, "series": ser3}, indent=1))
    line_figure(m_grid, ser3, "comparisons per aspect", "sparsity")

    # text summary -----------------------------------------------------------------------------------
    names = list(e1[0.8].keys())
    lines = [f"E1  Spearman ρ vs true quality, mean ± sd over {S1} seeds (ε = 0.1, 300 comparisons/aspect)",
             f"{'method':28s}" + "".join(f"{'γ=' + str(g):>16s}" for g in (0.0, 0.5, 0.8))]
    for k in names:
        lines.append(f"{k:28s}" + "".join(f"{e1[g][k][0]:>10.3f} ± {e1[g][k][1]:.2f}" for g in (0.0, 0.5, 0.8)))
    lines.append(f"EDR calibration (γ=0.8): ECE = {ece:.3f}")
    lines.append("\nE2  break-even (γ=0.8): " + json.dumps({k: [round(v, 3) for v in ser2[k][0]] for k in show}) + f"  eps={eps_grid}")
    lines.append("E3  sparsity    (γ=0.8): " + json.dumps({k: [round(v, 3) for v in ser3[k][0]] for k in show}) + f"  m={m_grid}")
    txt = "\n".join(lines); print(txt)
    (OUT / "summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
