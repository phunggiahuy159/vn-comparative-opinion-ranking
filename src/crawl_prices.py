"""Fetch product name, brand and listed price from each public product page's JSON-LD (schema.org Product).

Plain page GET (like a search engine), 5 s between requests (robots.txt Crawl-delay), resumable.
Input : data/tgdd_phone_urls*.txt     Output: data/tgdd_products.jsonl
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
UA = {"user-agent": "Mozilla/5.0 (academic crawler; HUST IT4868 course project; 1 req / 5 s)"}


def parse(html):
    out = {}
    for block in re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', html, re.S):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for d in data if isinstance(data, list) else data.get("@graph", [data]):
            if isinstance(d, dict) and d.get("@type") == "Product":
                offers = d.get("offers") or {}
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price = offers.get("price") or offers.get("lowPrice")
                brand = d.get("brand", {}).get("name") if isinstance(d.get("brand"), dict) else d.get("brand")
                if price and "price" not in out:
                    out = {"name": d.get("name"), "brand": brand, "price": float(price), "sku": d.get("sku")}
    if "price" not in out:  # fallback: first "price": number in page data
        m = re.search(r'"price":\s*"?([0-9]{6,})', html)
        if m:
            out["price"] = float(m.group(1))
    return out


def main(files):
    out_p = ROOT / "data" / "tgdd_products.jsonl"
    done = {json.loads(l)["url"] for l in open(out_p)} if out_p.exists() else set()
    urls = []
    for f in files:
        urls += [u.strip() for u in open(f) if u.strip()]
    urls = [u for u in dict.fromkeys(urls) if u not in done]
    with open(out_p, "a") as f:
        for n, u in enumerate(urls, 1):
            time.sleep(5)
            try:
                r = requests.get(u, headers=UA, timeout=30)
                rec = {"url": u, "slug": u.rstrip("/").split("/")[-1], "status": r.status_code,
                       **(parse(r.text) if r.status_code == 200 else {})}
            except requests.RequestException as e:
                rec = {"url": u, "slug": u.rstrip("/").split("/")[-1], "status": type(e).__name__}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{n}/{len(urls)}] {rec['slug']} {rec.get('price')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(ROOT / "data" / "tgdd_phone_urls.txt")])
