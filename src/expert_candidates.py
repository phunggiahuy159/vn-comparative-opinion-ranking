"""Candidate comparative sentences from recent VnExpress articles, for LLM extraction (src/llm_extract.py).

Keeps sentences from articles in the catalogue's era (default: 2025–2026) that contain a comparative cue and mention
either two catalogue phones, or one catalogue phone other than the article's own (title) phone, or the title phone's
predecessor/rival. Each candidate carries the title phone, which plays the role of the implicit subject ("THIS").
Output: data/expert_candidates.jsonl   Usage: python src/expert_candidates.py [--years 2025 2026]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from compare_rules import MODEL  # noqa: E402
from entity_link import Catalogue  # noqa: E402
from pipeline_real import build_products, link_title, load_prices, load_reviews  # noqa: E402

CUE = re.compile(r"hơn|kém|không bằng|thua|ngang|tương đương|vượt|nhỉnh|so với|bằng", re.I)
RIVAL = re.compile(r"\b(đối thủ|thế hệ trước|người tiền nhiệm)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2025", "2026"])
    args = ap.parse_args()
    model_re = re.compile(MODEL, re.I)
    products = build_products(load_reviews(), load_prices())
    cat = Catalogue(products)
    out, seen = [], set()
    for line in open(ROOT / "data" / "vnexpress_articles.jsonl"):
        a = json.loads(line)
        if not a.get("text") or not any(f"/{y}" in a.get("date", "") for y in args.years):
            continue
        this = link_title(cat, a["title"])
        this = this if this in products else None
        for sent in re.split(r"(?<=[.!?])\s+", a["text"]):
            sent = sent.strip()
            if not CUE.search(sent) or sent in seen:
                continue
            ms = {cat.link(m.group(0)) for m in model_re.finditer(sent)} & set(products)
            if len(ms) >= 2 or (this and ms - {this}) or (this and RIVAL.search(sent)):
                seen.add(sent)
                ctx = f"[Bài viết về: {products[this]['name']}] " if this else ""
                out.append({"id": f"vne-{len(out)}", "comment": ctx + sent[:600], "url": a["url"], "title_product": this,
                            "rule_tuples": []})
    with open(ROOT / "data" / "expert_candidates.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} candidate sentences -> data/expert_candidates.jsonl")


if __name__ == "__main__":
    main()
