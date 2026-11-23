"""Validation study for EDR v2: accuracy, ablations, parameter recovery, posterior calibration, value-for-money.

World (per seed): 40 phones, 6 brands, 2 aspects. Hedonic truth: q = μ_p + c·z(log price) + brand_q + noise.
Market expectation ê = c·z + brand_q. Sentiment: d = q − γ·ê, v = d (d≥0) or κ·d (d<0), P(+) = σ(α v + b).
Comparisons ~ Bradley–Terry(q), Zipf popularity, 30% brand-only mentions, flip rate ε (known = "measured").
Scenarios: EDT world (γ=.8, κ=2), absolute world (γ=0, κ=1: does EDR do harm?), sparse EDT world (100 comps).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import joint_model as J1  # noqa: E402
from edr import EDR, select_gamma  # noqa: E402
from ranking import bradley_terry, build_matrix, prank, sentiment_prior, zfuse  # noqa: E402

AS = ["BATTERY", "CAMERA"]


def world(seed, gamma=0.8, kappa=2.0, m=300, eps=0.1, brand_frac=0.3, n=40, nb=6, alpha=1.5):
    rng = np.random.default_rng(seed)
    prods = [f"p{i:02d}" for i in range(n)]
    brand = rng.integers(0, nb, n)
    logp = rng.normal(size=n); z = (logp - logp.mean()) / logp.std()
    price = dict(zip(prods, np.exp(16 + 0.5 * logp)))           # VND-ish scale; model z-scores log price
    bq = rng.normal(scale=0.4, size=(nb, len(AS)))
    c = np.array([0.7, 0.9])
    mu = rng.normal(scale=0.6, size=n)
    ehat = z[:, None] * c[None, :] + bq[brand]
    q = mu[:, None] + ehat + rng.normal(scale=0.5, size=(n, len(AS)))
    pop = rng.dirichlet(np.ones(n) * 0.5)
    groups = {f"b{k}": [prods[i] for i in range(n) if brand[i] == k] for k in range(nb)}
    comps = []
    for ai, a in enumerate(AS):
        for _ in range(m):
            i, j = rng.choice(n, 2, replace=False, p=pop)
            w, l = (i, j) if rng.random() < 1 / (1 + np.exp(-(q[i, ai] - q[j, ai]))) else (j, i)
            W, L = prods[w], prods[l]
            bl = rng.random() < brand_frac and brand[i] != brand[j]
            if bl:
                if w == j: W = f"group:b{brand[j]}"
                else: L = f"group:b{brand[j]}"
            if rng.random() < eps:
                W, L = L, W
            comps.append({"aspect": a, "winner": W, "loser": L, "brand_level": bl, "eps": eps})
    senti = {}
    for ai, a in enumerate(AS):
        for i in range(n):
            d = q[i, ai] - gamma * ehat[i, ai]
            v = d if d >= 0 else kappa * d
            k = rng.poisson(25 * pop[i] * n)
            pos = rng.binomial(k, 1 / (1 + np.exp(-(alpha * v))))
            senti[(prods[i], a)] = {"pos": int(pos), "neg": int(k - pos)}
    return dict(prods=prods, q=q, ehat=ehat, price=price, brand={prods[i]: f"b{brand[i]}" for i in range(n)},
                comps=comps, senti=senti, groups=groups)


def run(seed, **kw):
    W = world(seed, **kw)
    P, a, ai = W["prods"], "BATTERY", 0
    truth, vfm_true = W["q"][:, ai], W["q"][:, ai] - W["ehat"][:, ai]
    rho = lambda s: spearmanr([s[p] for p in P], truth).statistic
    ent = [c for c in W["comps"] if not c["brand_level"]]
    out = {}
    nodes, Wm = build_matrix(ent, a, P)
    bt = bradley_terry(nodes, Wm)
    out["price only"] = rho(W["price"])
    out["sentiment only"] = rho(dict(zip(nodes, sentiment_prior(nodes, a, W["senti"]))))
    out["pRank [Zhang 2010]"] = rho(prank(nodes, Wm, a, W["senti"]))
    out["BT + sentiment"] = rho(zfuse(bt, dict(zip(nodes, sentiment_prior(nodes, a, W["senti"])))))
    cov = {p: {f"brand_{b}": 1.0} for p, b in W["brand"].items()}
    s1, _ = J1.rank(P, a, W["comps"], W["senti"], price=W["price"], aspects=AS, covariates=cov, groups=W["groups"],
                    eps=kw.get("eps", 0.1))
    out["EDR v1"] = rho(s1)
    fitkw = dict(price=W["price"], brand=W["brand"], groups=W["groups"])
    extra = {}
    m, g, _ = select_gamma(P, AS, W["comps"], W["senti"], **fitkw)
    out["EDR (final: γ by held-out comps)"] = rho(m.quality(a))
    extra["gamma_cv"] = g
    for name, opts in {"  γ learned jointly": {}, "  γ = 0 (no price-relativity)": {"price_relative": False},
                       "  + loss aversion (γ by CV)": {"loss_aversion": True, "gamma": g},
                       "  − hedonic prior (γ by CV)": {"hedonic_prior": False, "gamma": g}}.items():
        mm = EDR(**opts).fit(P, AS, W["comps"], W["senti"], **fitkw)
        out[name] = rho(mm.quality(a))
        if name == "  + loss aversion (γ by CV)":
            extra["kappa_hat"] = mm.summary()["kappa(loss aversion)"]
    vf = m.value_for_money(a)
    extra["vfm_rho"] = spearmanr([vf[p] for p in P], vfm_true).statistic
    probs, labels = [], []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            probs.append(m.prob_better(P[i], P[j], a)); labels.append(float(truth[i] > truth[j]))
    probs, labels = np.array(probs), np.array(labels)
    extra["brier"] = float(np.mean((probs - labels) ** 2))
    bins = np.clip((probs * 10).astype(int), 0, 9)
    extra["ece"] = float(sum(abs(probs[bins == k].mean() - labels[bins == k].mean()) * (bins == k).mean()
                             for k in range(10) if (bins == k).any()))
    # value-for-money baseline: sentiment ratio (what a naive "good for the price" reading would use)
    extra["vfm_rho_sentiment"] = spearmanr(sentiment_prior(nodes, a, W["senti"]), vfm_true).statistic
    return out, extra


def main():
    lines = []
    scen = [("EDT world (γ=0.8, κ=2), 300 comps/aspect", dict(gamma=0.8, kappa=2.0, m=300)),
            ("Absolute world (γ=0, κ=1): does EDR do harm?", dict(gamma=0.0, kappa=1.0, m=300)),
            ("Moderate EDT world (γ=0.5, κ=2), 300 comps/aspect", dict(gamma=0.5, kappa=2.0, m=300)),
            ("Sparse EDT world (γ=0.8, κ=2), 100 comps/aspect", dict(gamma=0.8, kappa=2.0, m=100))]
    for title, kw in scen:
        res = [run(s, **kw) for s in range(20)]
        lines.append(f"\n=== {title} — Spearman ρ vs true quality (20 seeds) ===")
        for name in res[0][0]:
            v = np.array([r[0][name] for r in res])
            lines.append(f"  {name:30s} {v.mean():.3f} ± {v.std():.2f}")
        ex = {k: np.array([r[1][k] for r in res]) for k in res[0][1]}
        lines.append(f"  γ chosen by held-out comparisons = {ex['gamma_cv'].mean():.2f} ± {ex['gamma_cv'].std():.2f}   (true {kw['gamma']})")
        lines.append(f"  recovered κ̂ = {ex['kappa_hat'].mean():.2f} ± {ex['kappa_hat'].std():.2f}   (true {kw['kappa']})")
        lines.append(f"  value-for-money ρ: EDR {ex['vfm_rho'].mean():.3f}  vs sentiment ratio {ex['vfm_rho_sentiment'].mean():.3f}")
        lines.append(f"  P(A≻B) calibration: Brier {ex['brier'].mean():.3f}, ECE {ex['ece'].mean():.3f}")
    txt = "\n".join(lines)
    print(txt)
    (Path(__file__).parent.parent / "results" / "study_edr_v2.txt").write_text(txt)


if __name__ == "__main__":
    main()
