"""Joint Expectation–Disconfirmation Ranking (EDR) — our proposed model.

Grounding: Expectation–Disconfirmation Theory (Oliver, 1980): satisfaction = f(performance − expectation).
A latent aspect quality theta_{p,a} ("performance") explains two kinds of evidence mined from reviews:

  (1) comparative opinions  — Bradley–Terry, noise-aware, entity- OR set-level:
        d = theta_{S_w,a} − theta_{S_l,a},  theta_{S,a} = mean of theta over the products in side S
            (S = one product, or a set such as all phones of a brand: "hơn samsung")
        P(observed "w ≻ l") = (1 − eps) * sigmoid(d) + eps * sigmoid(−d)      (eps = extraction flip rate)
  (2) direct aspect sentiment (ABSA counts) — expectation-adjusted:
        P(review of p on a is positive) = sigmoid(alpha_a * theta_{p,a} − beta_a · x_p + b_a)
        x_p = expectation covariates (z-scored price, brand one-hot, launch year, …); beta_a is LEARNED,
        so we can report "how price-/brand-relative Vietnamese reviewers are" per aspect.

Hierarchical prior shares strength across aspects (sparse products): theta_{p,a} ~ N(mu_p, tau²), mu_p ~ N(0,1).
Fitted by MAP (L-BFGS, CPU). Comparisons make the quality scale identifiable; without them, quality and
expectation are confounded (see ablations in src/study_advanced.py).
"""
import numpy as np
import torch

torch.set_num_threads(4)
_logsig = torch.nn.functional.logsigmoid


def _design(products, price=None, covariates=None):
    """Build the expectation design matrix X (P×K) from price and/or a dict of per-product features."""
    cols, names = [], []
    if price:
        pr = np.array([price.get(p, np.nan) for p in products], dtype=float)
        pr = np.where(np.isnan(pr), np.nanmean(pr), pr)
        cols.append((pr - pr.mean()) / (pr.std() + 1e-9)); names.append("price")
    if covariates:
        keys = sorted({k for d in covariates.values() for k in d})
        for k in keys:
            v = np.array([covariates.get(p, {}).get(k, 0.0) for p in products], dtype=float)
            if v.std() > 0:
                cols.append((v - v.mean()) / v.std()); names.append(k)
    X = np.stack(cols, 1) if cols else np.zeros((len(products), 0))
    return torch.tensor(X, dtype=torch.float64), names


def _side(x, pi, groups):
    """Return a weight vector index list for one comparison side (product id or 'group:<name>')."""
    if isinstance(x, str) and x.startswith("group:"):
        mem = [pi[m] for m in groups.get(x[6:], []) if m in pi]
        return [(i, 1.0 / len(mem)) for i in mem] if mem else None
    return [(pi[x], 1.0)] if x in pi else None


def fit(products, aspects, comps, senti, price=None, covariates=None, groups=None,
        eps=0.0, learn_eps=False, tau=0.7, iters=400):
    P, A = len(products), len(aspects)
    pi = {p: i for i, p in enumerate(products)}
    ai = {a: i for i, a in enumerate(aspects)}
    groups = groups or {}
    X, xnames = _design(products, price, covariates)
    K = X.shape[1]

    rows = []
    for c in comps:
        if c["aspect"] not in ai:
            continue
        w, l = _side(c["winner"], pi, groups), _side(c["loser"], pi, groups)
        if w and l:
            rows.append((w, l, ai[c["aspect"]], c.get("conf", 1.0), c.get("type", "better") == "equal"))
    C = len(rows)
    Wm = torch.zeros(C, P, dtype=torch.float64); Lm = torch.zeros(C, P, dtype=torch.float64)
    for k, (w, l, *_r) in enumerate(rows):
        for i, v in w: Wm[k, i] += v
        for i, v in l: Lm[k, i] += v
    ca = torch.tensor([r[2] for r in rows], dtype=torch.long)
    cc = torch.tensor([r[3] for r in rows], dtype=torch.float64)
    ceq = torch.tensor([r[4] for r in rows], dtype=torch.bool)

    sp, sa, spos, sneg = [], [], [], []
    for (p, a), d in senti.items():
        if p in pi and a in ai and d.get("pos", 0) + d.get("neg", 0) > 0:
            sp.append(pi[p]); sa.append(ai[a]); spos.append(d["pos"]); sneg.append(d["neg"])
    sp, sa = torch.tensor(sp, dtype=torch.long), torch.tensor(sa, dtype=torch.long)
    spos, sneg = torch.tensor(spos, dtype=torch.float64), torch.tensor(sneg, dtype=torch.float64)

    theta = torch.zeros(P, A, dtype=torch.float64, requires_grad=True)
    mu = torch.zeros(P, dtype=torch.float64, requires_grad=True)
    log_alpha = torch.zeros(A, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(A, K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(A, dtype=torch.float64, requires_grad=True)
    eps_raw = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    params = [theta, mu, log_alpha, beta, b] + ([eps_raw] if learn_eps else [])
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe")

    def cur_eps():
        return 0.5 * torch.sigmoid(eps_raw) if learn_eps else torch.tensor(float(eps), dtype=torch.float64)

    def nll():
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        if C:
            th_c = theta.T[ca]                                   # C×P: theta of each comp's aspect
            d = (Wm * th_c).sum(1) - (Lm * th_c).sum(1)
            e = cur_eps()
            ll_b = torch.logaddexp(torch.log1p(-e) + _logsig(d), torch.log(e + 1e-12) + _logsig(-d)) if (learn_eps or eps > 0) else _logsig(d)
            ll_e = 0.5 * (_logsig(d) + _logsig(-d))
            loss = loss - (cc * torch.where(ceq, ll_e, ll_b)).sum()
        if len(sp):
            expect = (beta[sa] * X[sp]).sum(1) if K else 0.0
            eta = torch.exp(log_alpha[sa]) * theta[sp, sa] - expect + b[sa]
            loss = loss - (spos * _logsig(eta) + sneg * _logsig(-eta)).sum()
        loss = loss + ((theta - mu[:, None]) ** 2).sum() / (2 * tau ** 2) + (mu ** 2).sum() / 2
        loss = loss + (log_alpha ** 2).sum() / 2 + (beta ** 2).sum() / 8 + (b ** 2).sum() / 8
        if learn_eps:
            loss = loss + (eps_raw + 2.0) ** 2 / 2   # prior: eps ≈ 0.06
        loss.backward()
        return loss

    opt.step(nll)
    th = theta.detach().numpy()
    return {"theta": {(p, a): th[pi[p], ai[a]] for p in products for a in aspects},
            "beta": {a: dict(zip(xnames, beta.detach().numpy()[ai[a]])) for a in aspects},
            "alpha": dict(zip(aspects, np.exp(log_alpha.detach().numpy()))),
            "eps": float(cur_eps().detach()), "n_comps_used": C}


def rank(products, aspect, comps, senti, price=None, aspects=None, **kw):
    aspects = aspects or sorted({c["aspect"] for c in comps} | {a for _, a in senti} | {aspect})
    m = fit(products, aspects, comps, senti, price=price, **kw)
    return {p: m["theta"][(p, aspect)] for p in products}, m


def predict_comparison(model, winner, loser, aspect, groups=None):
    """P(winner ≻ loser) under the fitted model (for held-out comparison prediction)."""
    groups = groups or {}

    def val(x):
        if isinstance(x, str) and x.startswith("group:"):
            mem = [model["theta"][(m, aspect)] for m in groups.get(x[6:], []) if (m, aspect) in model["theta"]]
            return float(np.mean(mem)) if mem else 0.0
        return model["theta"].get((x, aspect), 0.0)
    return 1 / (1 + np.exp(-(val(winner) - val(loser))))


def bootstrap_ranks(products, aspect, comps, senti, price=None, B=200, seed=0, **kw):
    """Resample comparisons and (pos,neg) counts → 90% interval of each product's rank (1 = best)."""
    rng = np.random.default_rng(seed)
    ranks = []
    for _ in range(B):
        cb = [comps[i] for i in rng.integers(0, len(comps), len(comps))] if comps else []
        sb = {}
        for k, d in senti.items():
            n = d["pos"] + d["neg"]
            pos = rng.binomial(n, (d["pos"] + 0.5) / (n + 1)) if n else 0
            sb[k] = {"pos": int(pos), "neg": int(n - pos)}
        s, _ = rank(products, aspect, cb, sb, price=price, **kw)
        order = sorted(products, key=lambda p: -s[p])
        ranks.append([order.index(p) + 1 for p in products])
    ranks = np.array(ranks)
    return {p: (int(np.percentile(ranks[:, i], 5)), int(np.median(ranks[:, i])), int(np.percentile(ranks[:, i], 95)))
            for i, p in enumerate(products)}
