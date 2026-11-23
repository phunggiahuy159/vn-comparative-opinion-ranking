"""EDR v2 — Expectation–Disconfirmation Ranking with a hedonic expectation, prospect-theory value and Laplace posterior.

Generative story for product p, aspect a (all parameters interpretable):

  expectation    ê_{p,a} = x_p · c_a                       market's expected quality from observable cues
                                                           (x = z-price, brand one-hots): a hedonic curve, learned
  quality        θ_{p,a} ~ N(μ_p + ê_{p,a}, τ²)            empirical-Bayes prior: products with little evidence
                 μ_p ~ N(0, σ_μ²)                          shrink toward what their price/brand predicts
  disconfirm.    d_{p,a} = θ_{p,a} − γ_a · ê_{p,a}          γ_a ∈ [0,1] = price-relativity index of reviewers
  value          v(d) = d (d ≥ 0),  κ · d (d < 0)           prospect-theory loss aversion κ (smoothed)
  sentiment      #pos ~ Binomial(n, σ(α_a · v(d) + b_a))    ABSA counts from buyer reviews
  comparisons    P(observe S_w ≻ S_l) = (1−ε_s)σ(Δ) + ε_s σ(−Δ),  Δ = θ̄_{S_w,a} − θ̄_{S_l,a}
                 S = product or set (brand); ε_s = extraction flip rate of source s (measured on gold data)

Outputs: θ (absolute quality), value-for-money d* = θ − ê, γ_a, κ, c_a, and a Laplace (Gaussian) posterior over θ
giving P(θ_i > θ_j) and credible rank intervals.
"""
import math

import numpy as np
import torch

torch.set_num_threads(4)
_ls = torch.nn.functional.logsigmoid
DT = torch.float64


def _soft_value(d, kappa, s=6.0):
    """Smooth prospect-theory value: ≈ d for d>0, ≈ κ·d for d<0."""
    sp = torch.nn.functional.softplus
    return sp(s * d) / s - kappa * sp(-s * d) / s


def design(products, price=None, brand=None):
    cols, names = [], []
    if price:
        v = np.array([price.get(p, np.nan) for p in products], dtype=float)
        v = np.log(np.where(np.isnan(v), np.nanmedian(v), v))          # log-price: diminishing returns
        cols.append((v - v.mean()) / (v.std() + 1e-9)); names.append("log_price")
    if brand:
        bs = sorted({brand[p] for p in products if p in brand})
        for b in bs:
            cols.append(np.array([1.0 if brand.get(p) == b else 0.0 for p in products])); names.append(f"brand:{b}")
    X = np.stack(cols, 1) if cols else np.zeros((len(products), 0))
    return torch.tensor(X, dtype=DT), names


class EDR:
    def __init__(self, tau=0.6, sigma_mu=1.0, loss_aversion=False, hedonic_prior=True, price_relative=True, gamma=None):
        """gamma=None: learn γ_a (weakly identified!); gamma=float: fix it (use select_gamma() to choose it by
        held-out comparison likelihood — comparisons are not price-relative, so they referee the sentiment model)."""
        self.tau, self.sigma_mu = tau, sigma_mu
        self.loss_aversion, self.hedonic_prior, self.price_relative = loss_aversion, hedonic_prior, price_relative
        self.gamma_fixed = gamma

    # ------------------------------------------------------------------ data
    def _prep(self, products, aspects, comps, senti, price, brand, groups, source_eps):
        P, A = len(products), len(aspects)
        self.products, self.aspects = list(products), list(aspects)
        pi = {p: i for i, p in enumerate(products)}
        ai = {a: i for i, a in enumerate(aspects)}
        self.pi, self.ai = pi, ai
        self.X, self.xnames = design(products, price, brand)
        groups = groups or {}

        def side(x):
            if isinstance(x, str) and x.startswith("group:"):
                mem = [pi[m] for m in groups.get(x[6:], []) if m in pi]
                return [(i, 1 / len(mem)) for i in mem] if mem else None
            return [(pi[x], 1.0)] if x in pi else None

        rows = []
        for c in comps:
            if c["aspect"] in ai:
                w, l = side(c["winner"]), side(c["loser"])
                if w and l:
                    eps = (source_eps or {}).get(c.get("source", "default"), c.get("eps", 0.0))
                    rows.append((w, l, ai[c["aspect"]], c.get("conf", 1.0), c.get("type", "better") == "equal", eps))
        C = len(rows)
        self.Wm = torch.zeros(C, P, dtype=DT); self.Lm = torch.zeros(C, P, dtype=DT)
        for k, (w, l, *_r) in enumerate(rows):
            for i, v in w: self.Wm[k, i] += v
            for i, v in l: self.Lm[k, i] += v
        self.ca = torch.tensor([r[2] for r in rows], dtype=torch.long)
        self.cc = torch.tensor([r[3] for r in rows], dtype=DT)
        self.ceq = torch.tensor([r[4] for r in rows], dtype=torch.bool)
        self.ceps = torch.tensor([r[5] for r in rows], dtype=DT)
        sp_, sa_, pos, neg = [], [], [], []
        for (p, a), d in senti.items():
            if p in pi and a in ai and d.get("pos", 0) + d.get("neg", 0) > 0:
                sp_.append(pi[p]); sa_.append(ai[a]); pos.append(d["pos"]); neg.append(d["neg"])
        self.sp, self.sa = torch.tensor(sp_, dtype=torch.long), torch.tensor(sa_, dtype=torch.long)
        self.pos, self.neg = torch.tensor(pos, dtype=DT), torch.tensor(neg, dtype=DT)
        return P, A, self.X.shape[1]

    # ------------------------------------------------------------------ model
    def _unpack(self):
        th, mu, c, g, lk, la, b = self.params
        if not self.price_relative:
            gamma = torch.zeros_like(g)
        elif self.gamma_fixed is not None:
            gamma = torch.full_like(g, float(self.gamma_fixed))
        else:
            gamma = torch.sigmoid(g)
        kappa = torch.exp(lk) if self.loss_aversion else torch.ones((), dtype=DT)
        return th, mu, c, gamma, kappa, torch.exp(la), b

    def _nlp(self, theta=None):
        """Negative log posterior (theta can be overridden for the Laplace Hessian)."""
        th, mu, c, gamma, kappa, alpha, b = self._unpack()
        if theta is not None:
            th = theta
        ehat = self.X @ c.T if self.X.shape[1] else torch.zeros_like(th)          # P×A expectation
        loss = torch.zeros((), dtype=DT)
        if len(self.ca):
            thc = th.T[self.ca]
            dlt = (self.Wm * thc).sum(1) - (self.Lm * thc).sum(1)
            e = self.ceps
            llb = torch.where(e > 0, torch.logaddexp(torch.log1p(-e) + _ls(dlt), torch.log(e + 1e-12) + _ls(-dlt)), _ls(dlt))
            lle = 0.5 * (_ls(dlt) + _ls(-dlt))
            loss = loss - (self.cc * torch.where(self.ceq, lle, llb)).sum()
        if len(self.sp):
            d = th[self.sp, self.sa] - gamma[self.sa] * ehat[self.sp, self.sa]
            v = _soft_value(d, kappa)
            eta = alpha[self.sa] * v + b[self.sa]
            loss = loss - (self.pos * _ls(eta) + self.neg * _ls(-eta)).sum()
        prior_mean = mu[:, None] + (ehat if self.hedonic_prior else 0.0)
        loss = loss + ((th - prior_mean) ** 2).sum() / (2 * self.tau ** 2) + (mu ** 2).sum() / (2 * self.sigma_mu ** 2)
        loss = loss + (c ** 2).sum() / 2 + (self.params[3] ** 2).sum() / (2 * 1.5 ** 2)
        loss = loss + (self.params[4] ** 2) / (2 * 0.5 ** 2) + (self.params[5] ** 2).sum() / 2 + (b ** 2).sum() / 8
        return loss

    def fit(self, products, aspects, comps, senti, price=None, brand=None, groups=None, source_eps=None, iters=500):
        P, A, K = self._prep(products, aspects, comps, senti, price, brand, groups, source_eps)
        z = lambda *s: torch.zeros(*s, dtype=DT, requires_grad=True)
        self.params = [z(P, A), z(P), z(A, K), z(A), torch.tensor(math.log(1.5), dtype=DT, requires_grad=True), z(A), z(A)]
        opt = torch.optim.LBFGS(self.params, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-9)

        def closure():
            opt.zero_grad(); l = self._nlp(); l.backward(); return l
        opt.step(closure)
        self._laplace()
        return self

    def _laplace(self):
        """Gaussian approximation of p(θ | data, other params at MAP): Σ = H_θθ^{-1}."""
        th0 = self.params[0].detach().clone()
        f = lambda flat: self._nlp(flat.view_as(th0))
        H = torch.autograd.functional.hessian(f, th0.flatten())
        H = 0.5 * (H + H.T) + 1e-6 * torch.eye(H.shape[0], dtype=DT)
        self.mean = th0.numpy()
        self.cov = torch.linalg.inv(H).numpy()

    # ------------------------------------------------------------------ outputs
    def summary(self):
        _, _, c, gamma, kappa, alpha, b = self._unpack()
        c = c.detach().numpy()
        return {"gamma(price-relativity)": dict(zip(self.aspects, np.round(gamma.detach().numpy(), 3))),
                "kappa(loss aversion)": round(float(kappa.detach()), 3),
                "alpha": dict(zip(self.aspects, np.round(alpha.detach().numpy(), 3))),
                "hedonic_coef": {a: dict(zip(self.xnames, np.round(c[i], 3))) for i, a in enumerate(self.aspects)}}

    def quality(self, aspect):
        a = self.ai[aspect]
        return {p: float(self.mean[i, a]) for p, i in self.pi.items()}

    def value_for_money(self, aspect):
        """θ − ê: quality above/below what price & brand predict."""
        a = self.ai[aspect]
        c = self.params[2].detach()
        ehat = (self.X @ c.T).numpy() if self.X.shape[1] else np.zeros_like(self.mean)
        return {p: float(self.mean[i, a] - ehat[i, a]) for p, i in self.pi.items()}

    def prob_better(self, p, q, aspect):
        a, A = self.ai[aspect], len(self.aspects)
        i, j = self.pi[p] * A + a, self.pi[q] * A + a
        m = self.mean.flatten()
        var = self.cov[i, i] + self.cov[j, j] - 2 * self.cov[i, j]
        return float(0.5 * (1 + math.erf((m[i] - m[j]) / math.sqrt(2 * max(var, 1e-12)))))

    def rank_intervals(self, aspect, n=2000, seed=0, level=0.9):
        rng = np.random.default_rng(seed)
        a, A = self.ai[aspect], len(self.aspects)
        idx = [self.pi[p] * A + a for p in self.products]
        S = rng.multivariate_normal(self.mean.flatten()[idx], self.cov[np.ix_(idx, idx)], size=n)
        ranks = (-S).argsort(1).argsort(1) + 1
        lo, hi = np.percentile(ranks, [(1 - level) / 2 * 100, (1 + level) / 2 * 100], axis=0)
        med = np.median(ranks, 0)
        return {p: (int(lo[k]), int(med[k]), int(hi[k])) for k, p in enumerate(self.products)}


def heldout_comparison_ll(model, comps, aspect_filter=None):
    """Mean log-likelihood of held-out comparison directions under the fitted θ (entity-level only)."""
    ll, n = 0.0, 0
    for c in comps:
        if c["winner"] in model.pi and c["loser"] in model.pi and (aspect_filter is None or c["aspect"] == aspect_filter):
            if c["aspect"] not in model.ai or c.get("type", "better") == "equal":
                continue
            a = model.ai[c["aspect"]]
            d = model.mean[model.pi[c["winner"]], a] - model.mean[model.pi[c["loser"]], a]
            e = c.get("eps", 0.0)
            p = (1 - e) / (1 + math.exp(-d)) + e / (1 + math.exp(d))
            ll += math.log(max(p, 1e-12)); n += 1
    return ll / max(n, 1), n


def select_gamma(products, aspects, comps, senti, grid=(0.0, 0.25, 0.5, 0.75, 1.0), folds=5, seed=0, **fit_kw):
    """Choose the price-relativity γ by K-fold held-out comparison log-likelihood; refit on all data.
    Returns (fitted EDR, chosen γ, {γ: mean held-out ll})."""
    rng = np.random.default_rng(seed)
    ent_idx = [i for i, c in enumerate(comps) if not str(c["winner"]).startswith("group:")
               and not str(c["loser"]).startswith("group:")]
    rng.shuffle(ent_idx)
    fold_of = {i: k % folds for k, i in enumerate(ent_idx)}
    scores = {}
    for g in grid:
        lls = []
        for f in range(folds):
            train = [c for i, c in enumerate(comps) if fold_of.get(i, -1) != f]
            test = [comps[i] for i in ent_idx if fold_of[i] == f]
            m = EDR(gamma=g, **{k: v for k, v in fit_kw.items() if k in ("tau", "sigma_mu", "hedonic_prior", "loss_aversion")})
            m.fit(products, aspects, train, senti, **{k: v for k, v in fit_kw.items() if k in ("price", "brand", "groups", "source_eps")})
            lls.append(heldout_comparison_ll(m, test)[0])
        scores[g] = float(np.mean(lls))
    best = max(scores, key=scores.get)
    m = EDR(gamma=best, **{k: v for k, v in fit_kw.items() if k in ("tau", "sigma_mu", "hedonic_prior", "loss_aversion")})
    m.fit(products, aspects, comps, senti, **{k: v for k, v in fit_kw.items() if k in ("price", "brand", "groups", "source_eps")})
    return m, best, scores
