"""Rank real products per aspect with all rankers; evaluate against benchmarks when available.

Input : data/real_graph.json (from pipeline_real.py), optional data/benchmarks.csv
        benchmarks.csv columns: product_key,name,price_vnd,battery_gsmarena_h,camera_dxomark,perf_antutu,screen_dxomark
Output: results/real_rankings.json, results/real_eval.json; prints top-10 per aspect with rank intervals.
Without benchmarks it still reports held-out comparison accuracy (5-fold) — a ground-truth-free check.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import joint_model as J  # noqa: E402
from ranking import bradley_terry, build_matrix, prank, sentiment_prior  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH_COL = {"BATTERY": "battery_gsmarena_h", "CAMERA": "camera_dxomark", "PERFORMANCE": "perf_antutu", "SCREEN": "screen_dxomark"}
ASPECTS = ["BATTERY", "CAMERA", "PERFORMANCE", "SCREEN"]


def load(min_reviews):
    g = json.loads((ROOT / "data" / "real_graph.json").read_text())
    prods = {k: v for k, v in g["products"].items() if v["n_reviews"] >= min_reviews}
    senti = {}
    for k, v in g["senti"].items():
        p, a = k.split("|")
        if p in prods:
            senti[(p, a)] = {"pos": v["pos"], "neg": v["neg"]}
    groups = {}
    for k, v in g["products"].items():
        groups.setdefault(v["brand"], []).append(k)
    comps = [c for c in g["comps"]]
    # external / brand nodes stay as extra graph nodes for BT-style rankers
    return g["products"], prods, senti, comps, groups


def bench():
    p = ROOT / "data" / "benchmarks.csv"
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        for a, col in BENCH_COL.items():
            if r.get(col):
                out.setdefault(a, {})[r["product_key"]] = float(r[col])
    return out


def main(min_reviews=10):
    allp, prods, senti, comps, groups = load(min_reviews)
    keys = list(prods)
    price = {k: v["price"] for k, v in prods.items() if v.get("price")}
    # entity-level comps among kept products + groups (EDR uses groups; others drop them)
    ent = [c for c in comps if c["winner"] in prods and c["loser"] in prods]
    grp = [dict(c, winner=("group:" + c["winner"][6:]) if c["winner"].startswith("group:") else c["winner"],
                loser=("group:" + c["loser"][6:]) if c["loser"].startswith("group:") else c["loser"])
           for c in comps if (c["winner"] in prods or c["winner"].startswith("group:"))
           and (c["loser"] in prods or c["loser"].startswith("group:")) and c not in ent]
    print(f"products(>= {min_reviews} reviews)={len(keys)}  entity comps={len(ent)}  brand comps={len(grp)}  with price={len(price)}")
    B = bench()
    rankings, evals = {}, {}
    for a in ASPECTS:
        nodes, W = build_matrix(ent, a, keys)
        res = {"sentiment": dict(zip(nodes, sentiment_prior(nodes, a, senti))),
               "Bradley-Terry": bradley_terry(nodes, W),
               "pRank": prank(nodes, W, a, senti)}
        s, m = J.rank(keys, a, ent + grp, senti, price=price or None, aspects=ASPECTS,
                      groups={b: [k for k in ks if k in prods] for b, ks in groups.items()})
        res["EDR"] = s
        if price:
            res["price only"] = {k: price.get(k, np.nan) for k in keys}
        rankings[a] = {name: sorted(keys, key=lambda k: -np.nan_to_num(r[k], nan=-1e9)) for name, r in res.items()}
        evals[a] = {"beta": {k: float(v) for k, v in m["beta"][a].items()}}
        if a in B:
            common = [k for k in keys if k in B[a]]
            for name, r in res.items():
                x = [r[k] for k in common]; y = [B[a][k] for k in common]
                if len(common) >= 5:
                    evals[a][name] = {"n": len(common), "spearman": float(spearmanr(x, y).statistic),
                                      "kendall": float(kendalltau(x, y).statistic)}
        print(f"\n== {a}  (learned beta: {evals[a]['beta']})")
        for k in rankings[a]["EDR"][:8]:
            print(f"   {allp[k]['name'][:34]:34s} n_rev={allp[k]['n_reviews']:3d}  "
                  f"pos/neg={senti.get((k, a), {}).get('pos', 0)}/{senti.get((k, a), {}).get('neg', 0)}")
    (ROOT / "results" / "real_rankings.json").write_text(json.dumps(
        {a: {n: [allp[k]["name"] for k in r] for n, r in v.items()} for a, v in rankings.items()}, ensure_ascii=False, indent=1))
    (ROOT / "results" / "real_eval.json").write_text(json.dumps(evals, indent=2))
    # benchmark template for the team
    tpl = ROOT / "data" / "benchmarks_template.csv"
    with open(tpl, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_key", "name", "n_reviews", "price_vnd"] + list(BENCH_COL.values()) + ["source_urls"])
        for k, v in sorted(allp.items(), key=lambda kv: -kv[1]["n_reviews"]):
            w.writerow([k, v["name"], v["n_reviews"], v.get("price") or ""] + [""] * len(BENCH_COL) + [""])
    print(f"\nwrote {tpl}")


if __name__ == "__main__":
    main()
