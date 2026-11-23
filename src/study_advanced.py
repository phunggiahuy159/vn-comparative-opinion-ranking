"""Controlled study of the EDR model's components under realistic failure modes of review mining.

Simulated world (per seed): 40 phones, 6 brands, 2 aspects (BATTERY evaluated, CAMERA shares a product prior).
  - true quality q_{p,a}; price = mean_a q + noise (corr ≈ 0.6)
  - comparisons: 400 per aspect, pairs drawn by Zipf-like popularity, outcome ~ Bradley–Terry(q)
      * brand_frac of them name only the competitor's BRAND ("hơn samsung") → object = group:<brand>
      * extraction flips the direction with prob eps
  - direct sentiment counts: P(+) = sigmoid(q − lam·z(price) − brand_expectation[brand])
Baselines cannot link brand-only mentions to a product, so they drop them (as current pipelines must).
Metrics: Spearman ρ vs true q_BATTERY; held-out accuracy of predicting observed comparison directions (20%).
Usage: .venv/bin/python src/study_advanced.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import joint_model as J  # noqa: E402
from ranking import bradley_terry, build_matrix, prank, sentiment_prior, zfuse  # noqa: E402

ASPECTS = ["BATTERY", "CAMERA"]


def simulate(seed, eps=0.15, brand_frac=0.4, lam=0.8, brand_sd=0.6, n=40, m=400, n_brands=6):
    rng = np.random.default_rng(seed)
    prods = [f"p{i:02d}" for i in range(n)]
    brand = rng.integers(0, n_brands, n)
    mu = rng.normal(size=n)
    q = {a: mu + 0.6 * rng.normal(size=n) for a in ASPECTS}
    price = (q["BATTERY"] + q["CAMERA"]) / 2 + rng.normal(scale=1.0, size=n)
    zprice = (price - price.mean()) / price.std()
    bexp = rng.normal(scale=brand_sd, size=n_brands)
    pop = rng.dirichlet(np.ones(n) * 0.5)
    groups = {f"b{k}": [prods[i] for i in range(n) if brand[i] == k] for k in range(n_brands)}
    comps = []
    for a in ASPECTS:
        for _ in range(m):
            i, j = rng.choice(n, 2, replace=False, p=pop)
            w, l = (i, j) if rng.random() < 1 / (1 + np.exp(-(q[a][i] - q[a][j]))) else (j, i)
            obs_w, obs_l = prods[w], prods[l]
            brand_level = rng.random() < brand_frac and brand[i] != brand[j]
            if brand_level:  # reviewer of i names only j's brand
                if w == j:
                    obs_w = f"group:b{brand[j]}"
                else:
                    obs_l = f"group:b{brand[j]}"
            if rng.random() < eps:
                obs_w, obs_l = obs_l, obs_w
            comps.append({"aspect": a, "winner": obs_w, "loser": obs_l, "brand_level": brand_level})
    senti = {}
    for a in ASPECTS:
        for i in range(n):
            k = rng.poisson(30 * pop[i] * n)
            pp = 1 / (1 + np.exp(-(q[a][i] - lam * zprice[i] - bexp[brand[i]])))
            pos = rng.binomial(k, pp)
            senti[(prods[i], a)] = {"pos": int(pos), "neg": int(k - pos)}
    cov = {prods[i]: {f"brand_{brand[i]}": 1.0} for i in range(n)}
    return dict(prods=prods, q=q, price=dict(zip(prods, price)), comps=comps, senti=senti, groups=groups, cov=cov)


def bt_prob(s, w, l):
    return s[w] / (s[w] + s[l])


def run(seed, **kw):
    W = simulate(seed, **kw)
    prods, comps, senti, a = W["prods"], W["comps"], W["senti"], "BATTERY"
    rng = np.random.default_rng(1000 + seed)
    entity = [c for c in comps if not c["brand_level"]]
    brand_lv = [c for c in comps if c["brand_level"]]
    test_idx = set(rng.choice(len(entity), int(0.2 * len(entity)), replace=False).tolist())
    train_ent = [c for k, c in enumerate(entity) if k not in test_idx]
    test = [c for k, c in enumerate(entity) if k in test_idx and c["aspect"] == a]
    truth = W["q"][a]
    out = {}

    def score(name, s, prob=None):
        rho = spearmanr([s[p] for p in prods], truth).statistic
        acc = np.mean([prob(c["winner"], c["loser"]) > 0.5 for c in test]) if prob else np.nan
        out[name] = (rho, acc)

    nodes, Wm = build_matrix(train_ent, a, prods)
    bt = bradley_terry(nodes, Wm)
    score("Bradley-Terry", bt, lambda w, l: bt_prob(bt, w, l))
    score("pRank [Zhang 2010]", prank(nodes, Wm, a, senti))
    score("BT + sentiment", zfuse(bt, dict(zip(nodes, sentiment_prior(nodes, a, senti)))))
    score("price only", W["price"])

    def joint(name, use_brand_comps, covariates, learn_eps, eps=0.0):
        cs = train_ent + (brand_lv if use_brand_comps else [])
        s, m = J.rank(prods, a, cs, senti, price=W["price"], aspects=ASPECTS, covariates=covariates,
                      groups=W["groups"], learn_eps=learn_eps, eps=eps)
        score(name, s, lambda w, l: J.predict_comparison(m, w, l, a, W["groups"]) )
        return m

    joint("EDR: price expectation", False, None, False)
    joint("  + brand expectation", False, W["cov"], False)
    joint("  + brand-level comps", True, W["cov"], False)
    m = joint("  + noise-aware, eps learned", True, W["cov"], True)
    out["_eps_hat"] = (m["eps"], np.nan)
    joint("  + noise-aware, eps measured", True, W["cov"], False, eps=kw.get("eps", 0.15))
    score("sentiment only", dict(zip(nodes, sentiment_prior(nodes, a, senti))))
    return out


def table(title, res):
    lines = [f"\n=== {title} ===", f"{'method':30s}{'Spearman ρ':>14s}{'held-out acc':>15s}"]
    for name in res[0]:
        r = np.array([x[name][0] for x in res]); acc = np.array([x[name][1] for x in res])
        if name == "_eps_hat":
            lines.append(f"{'  (learned eps value)':30s}{r.mean():>9.3f}")
            continue
        accs = f"{np.nanmean(acc):.3f}" if np.isfinite(acc).any() else "—"
        lines.append(f"{name:30s}{r.mean():>9.3f}±{r.std():.2f}{accs:>13s}")
    return lines


if __name__ == "__main__":
    seeds = range(20)
    lines = []
    for m in (400, 150):
        for eps in (0.0, 0.15, 0.3):
            res = [run(s, eps=eps, m=m) for s in seeds]
            lines += table(f"{m} comparisons/aspect, flip rate eps={eps}, 40% brand-only mentions, lam=0.8", res)
    # break-even curve: how accurate must comparative extraction be?
    lines.append("\n=== Break-even: Spearman ρ vs extraction flip rate (400 comps, 20 seeds) ===")
    lines.append(f"{'eps':>6s}{'EDR full(meas.)':>18s}{'sentiment':>12s}{'price only':>12s}{'BT+sent':>10s}{'pRank':>8s}")
    for eps in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        res = [run(s, eps=eps, m=400) for s in range(12)]
        g = lambda k: np.mean([x[k][0] for x in res])
        lines.append(f"{eps:>6.2f}{g('  + noise-aware, eps measured'):>18.3f}{g('sentiment only'):>12.3f}{g('price only'):>12.3f}{g('BT + sentiment'):>10.3f}{g('pRank [Zhang 2010]'):>8.3f}")
    txt = "\n".join(lines)
    print(txt)
    (Path(__file__).parent.parent / "results" / "study_advanced.txt").write_text(txt)
