"""Aspect-specific product ranking from comparative opinions (course L05 link analysis + L06 opinion mining).

Input: comparison records (one per extracted quintuple, after entity linking):
    {"aspect": "BATTERY", "winner": "galaxy-a50", "loser": "oppo-f11", "type": "better"|"equal", "conf": 0.9}
plus optional per-product aspect sentiment counts from (non-comparative) ABSA:
    {("galaxy-a50", "BATTERY"): {"pos": 120, "neg": 30}}

Rankers (all return {product: score}, higher = better):
  win_rate      — wins / (wins + losses), Laplace-smoothed
  bradley_terry — MLE of P(i beats j) = s_i / (s_i + s_j) via MM updates (Hunter 2004)
  pagerank      — loser -> winner edges ("a comparison is a vote", like a hyperlink), power iteration
  ppr           — personalized PageRank: teleport ∝ smoothed positive-sentiment prior from ABSA,
                  fusing the ~90% direct opinions with the ~10% comparative ones
  sentiment     — smoothed positive ratio only (no graph)
Evaluation: Spearman rho / Kendall tau / NDCG@k against external benchmark scores (DXOMARK, GSMArena, AnTuTu…).
"""
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr


def build_matrix(comps, aspect, nodes=None):
    comps = [c for c in comps if c["aspect"] == aspect]
    if nodes is None:
        nodes = sorted({c["winner"] for c in comps} | {c["loser"] for c in comps})
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))  # W[i, j] = weight of "i beats j"
    for c in comps:
        if c["winner"] not in idx or c["loser"] not in idx:
            continue
        w, l, conf = idx[c["winner"]], idx[c["loser"]], c.get("conf", 1.0)
        if c.get("type", "better") == "equal":
            W[w, l] += 0.5 * conf
            W[l, w] += 0.5 * conf
        else:
            W[w, l] += conf
    return nodes, W


def win_rate(nodes, W, alpha=1.0):
    wins, losses = W.sum(1), W.sum(0)
    return dict(zip(nodes, (wins + alpha) / (wins + losses + 2 * alpha)))


def bradley_terry(nodes, W, iters=500, prior=0.1, tol=1e-9):
    n = len(nodes)
    # small virtual comparisons against every other node keep the MLE finite on sparse/disconnected graphs
    W = W + prior * (1 - np.eye(n)) / max(n - 1, 1)
    N = W + W.T
    s = np.ones(n)
    for _ in range(iters):
        denom = (N / (s[:, None] + s[None, :])).sum(1)
        s_new = W.sum(1) / np.maximum(denom, 1e-12)
        s_new /= s_new.sum()
        if np.abs(s_new - s).max() < tol:
            s = s_new
            break
        s = s_new
    return dict(zip(nodes, s))


def pagerank(nodes, W, d=0.85, teleport=None, iters=200, tol=1e-10):
    """Edges loser -> winner with weight W[winner, loser]; dangling mass goes to the teleport vector."""
    n = len(nodes)
    A = W.T  # A[loser, winner]
    out = A.sum(1)
    v = np.full(n, 1 / n) if teleport is None else teleport / teleport.sum()
    r = v.copy()
    for _ in range(iters):
        flow = np.zeros(n)
        nz = out > 0
        flow += (r[nz, None] * A[nz] / out[nz, None]).sum(0)
        flow += r[~nz].sum() * v  # dangling nodes (never lost a comparison)
        r_new = (1 - d) * v + d * flow
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return dict(zip(nodes, r))


def degree_normalized(scores: dict, nodes, W, eps=1.0):
    """Divide a stationary score by #comparisons the product took part in (removes popularity bias)."""
    deg = W.sum(0) + W.sum(1)
    return {p: scores[p] / (deg[i] + eps) for i, p in enumerate(nodes)}


def random_walker(nodes, W, p=0.75, teleport=None, d=0.95, iters=500, tol=1e-12):
    """Callaghan-Mucha-Porter random-walker ranking: a walker at product i picks one of i's comparisons
    uniformly and moves to its winner w.p. p, to its loser w.p. 1-p. Winners keep mass via self-loops,
    so the walk is not dominated by how often a product is mentioned. Score = stationary mass / degree."""
    n = len(nodes)
    N = W + W.T  # comparisons per pair
    deg = N.sum(1)
    T = np.zeros((n, n))
    for i in range(n):
        if deg[i] == 0:
            continue
        for j in np.nonzero(N[i])[0]:
            # comparisons between i and j: W[i,j] won by i, W[j,i] won by j
            T[i, j] += (W[j, i] * p + W[i, j] * (1 - p)) / deg[i]  # move to j if j won (p) or i won (1-p)
            T[i, i] += (W[i, j] * p + W[j, i] * (1 - p)) / deg[i]  # stay if i won (p) or j won (1-p)
    T[deg == 0, :] = 0
    v = np.full(n, 1 / n) if teleport is None else teleport / teleport.sum()
    r = v.copy()
    for _ in range(iters):
        r_new = (1 - d) * v + d * (r @ T + r[deg == 0].sum() * v)
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return dict(zip(nodes, r))


def sentiment_prior(nodes, aspect, senti, a=2.0, b=2.0):
    """Beta-smoothed positive ratio per product for one aspect."""
    return np.array([(senti.get((p, aspect), {}).get("pos", 0) + a) /
                     (senti.get((p, aspect), {}).get("pos", 0) + senti.get((p, aspect), {}).get("neg", 0) + a + b)
                     for p in nodes])


def ppr(nodes, W, aspect, senti, d=0.85, gamma=2.0):
    prior = sentiment_prior(nodes, aspect, senti) ** gamma
    return pagerank(nodes, W, d=d, teleport=prior)


def prank(nodes, W, aspect, senti, d=0.85, alpha=1.0, iters=500, tol=1e-12, use_node_weight=True):
    """pRank — reproduction of Zhang, Narayanan & Choudhary (WOSN 2010), "Voice of the Customers".

    Per feature f: edge Pj -> Pi exists if some comparative sentence says Pi is better than Pj;
    its weight is PC(Pi,Pj) / NC(Pi,Pj) (#positive / #negative comparatives of Pi vs Pj).
    Node weight Wv(P) = PS / NS (#positive / #negative subjective sentences about P on f).
        pRank(P) = [(1-d) + d * sum_{Pi -> P} pRank(Pi) * Ce(Pi, P)] * Cv(P)
        Ce(Pi, P) = We(Pi, P) / sum_k We(Pi, Pk)      (edge-weight contributor)
        Cv(P)     = Wv(P) / sum_t Wv(Pt)              (node-weight contributor)
    The paper's ratios are undefined when NC = 0 or NS = 0, so we add-alpha smooth both (our choice).
    """
    n = len(nodes)
    We = np.zeros((n, n))  # We[j, i] = weight of edge Pj -> Pi
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > 0:  # Pi beat Pj at least once
                We[j, i] = (W[i, j] + alpha) / (W[j, i] + alpha)
    out = We.sum(1, keepdims=True)
    Ce = np.divide(We, out, out=np.zeros_like(We), where=out > 0)
    if use_node_weight:
        wv = np.array([(senti.get((p, aspect), {}).get("pos", 0) + alpha) /
                       (senti.get((p, aspect), {}).get("neg", 0) + alpha) for p in nodes])
    else:
        wv = np.ones(n)
    Cv = wv / wv.sum()
    r = np.full(n, 1 / n)
    for _ in range(iters):
        r_new = ((1 - d) + d * (Ce.T @ r)) * Cv
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return dict(zip(nodes, r))


def ndcg_at_k(pred: dict, truth: dict, k=10):
    common = [p for p in pred if p in truth]
    order = sorted(common, key=lambda p: -pred[p])[:k]
    ideal = sorted(common, key=lambda p: -truth[p])[:k]
    rel = {p: r for r, p in enumerate(sorted(common, key=lambda p: truth[p]), 1)}  # graded by benchmark rank
    dcg = sum(rel[p] / np.log2(i + 2) for i, p in enumerate(order))
    idcg = sum(rel[p] / np.log2(i + 2) for i, p in enumerate(ideal))
    return dcg / idcg if idcg else float("nan")


def evaluate(pred: dict, truth: dict, k=10):
    common = [p for p in pred if p in truth]
    x, y = [pred[p] for p in common], [truth[p] for p in common]
    return {"n": len(common), "spearman": spearmanr(x, y).statistic, "kendall": kendalltau(x, y).statistic,
            f"ndcg@{k}": ndcg_at_k(pred, truth, k)}


def bootstrap_ci(pred, truth, metric="spearman", B=1000, seed=0):
    rng = np.random.default_rng(seed)
    common = [p for p in pred if p in truth]
    vals = []
    for _ in range(B):
        s = rng.choice(common, len(common), replace=True)
        x, y = [pred[p] for p in s], [truth[p] for p in s]
        f = spearmanr if metric == "spearman" else kendalltau
        vals.append(f(x, y).statistic)
    return np.nanpercentile(vals, [2.5, 97.5])


def _joint():
    import joint_model
    return joint_model


def zfuse(*scores, log=True):
    """Late fusion: average of z-scored (log-)scores."""
    nodes = list(scores[0])
    zs = []
    for s in scores:
        v = np.array([s[p] for p in nodes], dtype=float)
        v = np.log(np.maximum(v, 1e-12)) if log else v
        zs.append((v - v.mean()) / (v.std() + 1e-12))
    return dict(zip(nodes, np.mean(zs, 0)))


def _demo(seed=42, lam=0.0, m=400, price_noise=0.7):
    """Synthetic check. Hidden quality q (= benchmark). Comparisons ~ Bradley-Terry(q) with Zipf-like popularity.
    Direct sentiment is PRICE-RELATIVE: logit = q - lam * price, price = q + noise (expensive phones are better
    on average). lam=0: sentiment is an unbiased quality signal; lam -> 1: sentiment only says "good for the price"."""
    rng = np.random.default_rng(seed)
    out = {}
    n = 30
    nodes = [f"p{i:02d}" for i in range(n)]
    q = rng.normal(size=n)  # hidden "benchmark" quality
    comps, senti = [], defaultdict(lambda: {"pos": 0, "neg": 0})
    pop = rng.dirichlet(np.ones(n) * 0.5)  # popularity skew: few phones get most comparisons
    price = q + rng.normal(scale=price_noise, size=n)
    for _ in range(m):
        i, j = rng.choice(n, 2, replace=False, p=pop)
        pi = 1 / (1 + np.exp(-(q[i] - q[j])))
        w, l = (i, j) if rng.random() < pi else (j, i)
        comps.append({"aspect": "BATTERY", "winner": nodes[w], "loser": nodes[l], "conf": 1.0})
    for i in range(n):
        k = rng.poisson(40 * pop[i] * n)
        pos = rng.binomial(k, 1 / (1 + np.exp(-(q[i] - lam * price[i]))))
        senti[(nodes[i], "BATTERY")] = {"pos": int(pos), "neg": int(k - pos)}
    truth = dict(zip(nodes, q))
    nodes_, W = build_matrix(comps, "BATTERY", nodes)
    for name, f in [("win_rate", lambda: win_rate(nodes_, W)), ("bradley_terry", lambda: bradley_terry(nodes_, W)),
                    ("pagerank", lambda: pagerank(nodes_, W)),
                    ("pagerank/deg", lambda: degree_normalized(pagerank(nodes_, W), nodes_, W)),
                    ("ppr", lambda: ppr(nodes_, W, "BATTERY", senti)),
                    ("ppr/deg", lambda: degree_normalized(ppr(nodes_, W, "BATTERY", senti), nodes_, W)),
                    ("random_walker", lambda: degree_normalized(random_walker(nodes_, W), nodes_, W)),
                    ("pRank[Zhang10]", lambda: prank(nodes_, W, "BATTERY", senti)),
                    ("pRank-edges", lambda: prank(nodes_, W, "BATTERY", senti, use_node_weight=False)),
                    ("sentiment", lambda: dict(zip(nodes_, sentiment_prior(nodes_, "BATTERY", senti)))),
                    ("price only", lambda: dict(zip(nodes, price))),
                    ("JOINT (ours)", lambda: _joint().rank(nodes_, "BATTERY", comps, senti, dict(zip(nodes, price)))[0]),
                    ("JOINT -price", lambda: _joint().rank(nodes_, "BATTERY", comps, senti, {})[0]),
                    ("JOINT -comps", lambda: _joint().rank(nodes_, "BATTERY", [], senti, dict(zip(nodes, price)))[0]),
                    ("BT+sentiment", lambda: zfuse(bradley_terry(nodes_, W),
                                                   dict(zip(nodes_, sentiment_prior(nodes_, "BATTERY", senti)))))]:
        out[name] = evaluate(f(), truth)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    lams = [0.0, 0.5, 0.8, 1.0]
    for pn, desc in [(0.7, "strong price-quality link, corr(price,q)~0.82"), (1.5, "weak price-quality link, corr(price,q)~0.55")]:
        res = {lam: [_demo(seed, lam=lam, price_noise=pn) for seed in range(30)] for lam in lams}
        print(f"\n=== Synthetic study ({desc}): 30 products, 400 comparisons, Zipf popularity, 30 seeds ===")
        print("Spearman rho vs hidden quality (mean ± sd); lam = how price-relative direct sentiment is.")
        print(f"{'ranker':16s}" + "".join(f"lam={l:<12}" for l in lams))
        for name in res[lams[0]][0]:
            row = []
            for lam in lams:
                v = np.array([r[name]["spearman"] for r in res[lam]])
                row.append(f"{v.mean():.3f}±{v.std():.2f}  ")
            print(f"{name:16s}" + "".join(f"{c:16s}" for c in row))
