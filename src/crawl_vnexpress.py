"""Crawl expert phone articles (reviews, comparisons) from VnExpress — the head-to-head comparison source.

Buyer reviews mostly compare against the reviewer's previous phone or a brand; journalists compare competing
models directly ("Galaxy S25 Ultra zoom xa hơn iPhone 16 Pro Max"). VCOM (VLSP 2023) was built from this kind of text.
robots.txt allows crawling; 5 s between requests; only article title/date/body text are stored.

Output: data/vnexpress_articles.jsonl
Usage : .venv/bin/python src/crawl_vnexpress.py [--pages 60]
"""
import argparse
import html as H
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
UA = {"user-agent": "Mozilla/5.0 (academic crawler; HUST IT4868 course project; 1 req / 5 s)"}
LISTS = ["https://vnexpress.net/khoa-hoc-cong-nghe/thiet-bi", "https://vnexpress.net/cong-nghe/san-pham/dien-thoai"]
PHONE = re.compile(r"iphone|galaxy|samsung|xiaomi|redmi|poco|oppo|vivo|realme|pixel|honor|huawei|nokia|vsmart|bphone|"
                   r"smartphone|điện thoại|dien-thoai|zenfone|motorola|nothing-phone|find-x|reno", re.I)


def get(url):
    time.sleep(5)
    try:
        r = requests.get(url, headers=UA, timeout=30)
        return r.text if r.status_code == 200 else None
    except requests.RequestException:
        return None


def article(url):
    h = get(url)
    if not h:
        return None
    title = re.search(r'<h1 class="title-detail[^"]*">(.*?)</h1>', h, re.S)
    date = re.search(r'<span class="date">(.*?)</span>', h, re.S)
    paras = re.findall(r'<p class="Normal"[^>]*>(.*?)</p>', h, re.S)
    clean = lambda s: H.unescape(re.sub(r"<[^>]+>", "", s)).strip()
    body = "\n".join(clean(p) for p in paras if clean(p))
    return {"url": url, "title": clean(title.group(1)) if title else "", "date": clean(date.group(1)) if date else "",
            "text": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=60)
    args = ap.parse_args()
    out_p = ROOT / "data" / "vnexpress_articles.jsonl"
    done = {json.loads(l)["url"] for l in open(out_p)} if out_p.exists() else set()
    links = []
    for base in LISTS:
        for p in range(1, args.pages + 1):
            h = get(base if p == 1 else f"{base}-p{p}")
            if not h:
                break
            found = re.findall(r'href="(https://vnexpress\.net/[a-z0-9-]+-\d{7}\.html)"', h)
            new = [u for u in dict.fromkeys(found) if PHONE.search(u) and u not in done and u not in links]
            links += new
            print(f"list {base.split('/')[-1]} p{p}: +{len(new)} (total {len(links)})", flush=True)
    with open(out_p, "a") as f:
        for n, u in enumerate(links, 1):
            a = article(u)
            if a and a["text"]:
                f.write(json.dumps(a, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{n}/{len(links)}] {u.split('/')[-1][:70]} {len(a['text']) if a else 'ERR'}", flush=True)


if __name__ == "__main__":
    main()
