"""End-to-end pipeline on crawled reviews (CPU only).

reviews (data/tgdd_reviews*.jsonl) ─► PhoBERT ABSA (cached) ─► per-(product, aspect) pos/neg counts
                                   └► comparative extraction (rules now; LLM/PhoBERT later) ─► entity linking
                                      ─► comparisons {winner, loser, aspect, type, conf}
Writes data/real_graph.json (products, senti, comps) + results/real_pipeline_stats.json.
Usage: .venv/bin/python src/pipeline_real.py [--extractor rules|llm]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_rules import extract as rule_extract  # noqa: E402
from entity_link import Catalogue  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RANKED = ["BATTERY", "CAMERA", "PERFORMANCE", "SCREEN", "DESIGN", "PRICE", "GENERAL", "FEATURES"]


def base_name(slug):
    s = re.sub(r"-\d+gb.*$", "", slug)
    s = re.sub(r"-(den|trang|xanh|hong|vang|tim|bac|xam|do|cam|titan.*|molly)$", "", s)
    return s.replace("-", " ")


def load_reviews():
    rs = []
    for f in sorted((ROOT / "data").glob("tgdd_reviews*.jsonl")):
        rs += [json.loads(l) for l in open(f)]
    seen, out = set(), []
    for r in rs:
        if r["commentId"] not in seen and (r.get("content") or "").strip():
            seen.add(r["commentId"]); out.append({**r, "source": "tgdd"})
    # second source: CellphoneS reviews, already mapped to the thegioididong modelId (data/cps_targets.json);
    # they inherit that model's thegioididong slug so prices and spec sheets line up.
    cps = ROOT / "data" / "cps_reviews.jsonl"
    if cps.exists():
        slug_of = {}
        for r in out:
            slug_of.setdefault(str(r["modelId"]), r["slug"])
        for l in open(cps):
            c = json.loads(l)
            cid = f"cps-{c['id']}"
            if cid in seen or not (c.get("content") or "").strip() or str(c["modelId"]) not in slug_of:
                continue
            seen.add(cid)
            out.append({"commentId": cid, "content": c["content"], "score": c.get("rating"), "modelId": c["modelId"],
                        "slug": slug_of[str(c["modelId"])], "createdDate": c.get("created_at"), "source": "cellphones"})
    return out


def load_prices():
    """slug → listed price; the spec-sheet crawl (newer, more pages) takes precedence over the price-only crawl."""
    prices = {}
    for name in ("tgdd_products_specs.jsonl", "tgdd_products.jsonl"):
        p = ROOT / "data" / name
        if p.exists():
            for l in open(p):
                d = json.loads(l)
                if d.get("price"):
                    prices.setdefault(d["slug"], d["price"])
    return prices


def build_products(reviews, prices):
    by_model = collections.defaultdict(list)
    for r in reviews:
        by_model[str(r["modelId"])].append(r)
    products = {}
    for mid, rs in by_model.items():
        slugs = collections.Counter(r["slug"] for r in rs)
        name = base_name(min(slugs, key=len))
        brand = name.split()[0]
        price = min([prices[s] for s in slugs if s in prices], default=None)
        products[mid] = {"name": name, "brand": brand, "n_reviews": len(rs), "price": price, "slugs": sorted(slugs)}
    return products


def absa_counts(reviews, cache_path):
    cache = {}
    if cache_path.exists():
        cache = {json.loads(l)["commentId"]: json.loads(l)["absa"] for l in open(cache_path)}
    todo = [r for r in reviews if r["commentId"] not in cache]
    if todo:
        from absa_infer import ABSAPredictor
        pred = ABSAPredictor(ROOT / "results" / "absa_phobert-base-v2", device="cpu")
        outs = pred.predict([r["content"] for r in todo])
        with open(cache_path, "a") as f:
            for r, o in zip(todo, outs):
                cache[r["commentId"]] = {a: [lab, round(p, 3)] for a, (lab, p) in o.items()}
                f.write(json.dumps({"commentId": r["commentId"], "absa": cache[r["commentId"]]}, ensure_ascii=False) + "\n")
    senti = collections.defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0})
    for r in reviews:
        for a, (lab, _) in cache[r["commentId"]].items():
            senti[(str(r["modelId"]), a)][{"Positive": "pos", "Negative": "neg", "Neutral": "neu"}[lab]] += 1
    return senti, cache


def comparisons_rules(reviews, cat):
    comps, raw, unresolved = [], 0, collections.Counter()
    for r in reviews:
        this = str(r["modelId"])
        for t in rule_extract(r["content"]):
            if t["label"].startswith("SUP") or not t["object"]:
                continue
            raw += 1
            subj = this if t["subject"] == "THIS" else cat.link(t["subject"])
            obj = cat.link(t["object"])
            if not subj or not obj or subj == obj:
                unresolved[t["object"]] += 1
                continue
            if t["label"] == "EQL":
                w, l, typ = subj, obj, "equal"
            else:
                w, l, typ = (subj, obj, "better") if t["label"] == "COM+" else (obj, subj, "better")
            comps.append({"winner": w, "loser": l, "aspect": t["aspect"], "type": typ, "conf": 1.0,
                          "commentId": r["commentId"], "evidence": t["clause"][:200]})
    return comps, raw, unresolved


def link_title(cat, title):
    """Implicit subject of an expert article = first product mention in its title that links to something."""
    from compare_rules import MODEL
    for m in re.finditer(MODEL, (title or "").lower()):
        e = cat.link(m.group(0))
        if e:
            return e
    return None


def comparisons_articles(articles, cat):
    """Expert comparisons from VnExpress articles (rules for now; swap in LLM tuples when available).
    Subject THIS → the title's product; clauses are de-duplicated per article; source='expert'."""
    comps, seen = [], set()
    for a in articles:
        this = link_title(cat, a["title"])
        for t in rule_extract(a["text"]):
            if t["label"].startswith("SUP") or not t["object"]:
                continue
            key = (a["url"], t["clause"], t["object"])
            if key in seen:
                continue
            seen.add(key)
            subj = this if t["subject"] == "THIS" else cat.link(t["subject"])
            obj = cat.link(t["object"])
            if not subj or not obj or subj == obj:
                continue
            w, l = (subj, obj) if t["label"] in ("COM+", "EQL") else (obj, subj)
            comps.append({"winner": w, "loser": l, "aspect": t["aspect"], "conf": 1.0, "source": "expert",
                          "type": "equal" if t["label"] == "EQL" else "better", "evidence": t["clause"][:200], "url": a["url"]})
    return comps


def comparisons_expert_llm(cat, path=ROOT / "results" / "llm_expert_tuples.jsonl"):
    """Expert comparisons from LLM quintuples on VnExpress sentences (src/expert_candidates.py → src/llm_extract.py).
    THIS = the article's title phone; COM+/COM-/EQL become graph edges with source='expert'."""
    if not path.exists():
        return []
    comps, seen = [], set()
    for line in open(path):
        r = json.loads(line)
        for t in r.get("llm_tuples") or []:
            lab = t.get("label")
            if lab not in ("COM+", "COM-", "EQL") or not t.get("object"):
                continue
            subj = r.get("title_product") if str(t.get("subject", "")).upper() == "THIS" else cat.link(str(t.get("subject", "")))
            obj = cat.link(str(t["object"]))
            if not subj or not obj or subj == obj:
                continue
            aspect = t.get("aspect") if t.get("aspect") in RANKED else "GENERAL"
            w, l = (subj, obj) if lab in ("COM+", "EQL") else (obj, subj)
            key = (r["id"], w, l, aspect)
            if key in seen:
                continue
            seen.add(key)
            comps.append({"winner": w, "loser": l, "aspect": aspect, "conf": 1.0, "source": "expert",
                          "type": "equal" if lab == "EQL" else "better", "evidence": r["comment"][:200], "url": r.get("url")})
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="rules")
    args = ap.parse_args()
    reviews = load_reviews()
    prices = load_prices()
    products = build_products(reviews, prices)
    cat = Catalogue(products)
    senti, _ = absa_counts(reviews, ROOT / "data" / "real_absa_cache.jsonl")
    comps, raw, unresolved = comparisons_rules(reviews, cat)
    expert = comparisons_expert_llm(cat)
    comps += expert

    kinds = collections.Counter(("cat" if c["winner"] in products else c["winner"].split(":")[0]) + "-" +
                                ("cat" if c["loser"] in products else c["loser"].split(":")[0]) for c in comps)
    stats = {"reviews": len(reviews), "reviews_by_source": dict(collections.Counter(r["source"] for r in reviews)),
             "products": len(products),
             "products_with_price": sum(p["price"] is not None for p in products.values()),
             "products_>=20_reviews": sum(p["n_reviews"] >= 20 for p in products.values()),
             "rule_tuples_with_object": raw, "linked_comparisons": len(comps), "expert_llm_comparisons": len(expert), "edge_kinds": dict(kinds),
             "by_aspect": dict(collections.Counter(c["aspect"] for c in comps)),
             "top_unresolved_objects": unresolved.most_common(10),
             "absa_aspect_mentions": {a: sum(v["pos"] + v["neg"] for (p, aa), v in senti.items() if aa == a) for a in RANKED}}
    (ROOT / "results" / "real_pipeline_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    graph = {"products": products, "senti": {f"{p}|{a}": v for (p, a), v in senti.items()}, "comps": comps}
    (ROOT / "data" / "real_graph.json").write_text(json.dumps(graph, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    for c in comps[:12]:
        print(f"  {c['aspect']:12s} {products.get(c['winner'], {}).get('name', c['winner'])} > "
              f"{products.get(c['loser'], {}).get('name', c['loser'])}   «{c['evidence'][:90]}»")


if __name__ == "__main__":
    main()
