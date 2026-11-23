"""Crawl buyer reviews per phone from cellphones.com.vn (second review source).

robots.txt allows product pages (it disallows query-string URLs, which we never request). Each product's public review page
(<slug>/review) is rendered in a headless browser; the first page of reviews comes with the page, further pages are
reached by clicking the page's own "Xem thêm đánh giá" button. We record the review data the page receives.
Politeness: 5 s between page actions, one tab, resumable.
Privacy: reviewer names / customer ids are NOT stored — only text, star rating, per-aspect ratings, dates, purchase flag.

Step 1: --targets   public product sitemaps → data/cps_sitemap_urls.txt; match to our thegioididong models → data/cps_targets.json
Step 2: (default)   crawl reviews → data/cps_reviews.jsonl, data/cps_crawl_state.json
Usage : .venv/bin/python src/crawl_cellphones.py --targets ; .venv/bin/python src/crawl_cellphones.py [--max_pages 40]
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (academic crawler; HUST IT4868 course project; 1 action / 5 s)"


SITEMAP = "https://cellphones.com.vn/sitemap/sitemap_index.xml"
NOT_PHONE = re.compile(r"cu-|op-lung|mieng|dan-|bao-da|trung-bay|buyback|kinh-|sac-|cap-|tai-nghe|watch|buds|ban-phim|day-|combo|goi-|bao-hanh")


def _norm(slug):
    s = slug.lower().replace(".html", "")
    s = re.sub(r"^dien-thoai-", "", s)
    s = re.sub(r"-\d+gb.*$", "", s)
    s = re.sub(r"-(5g|4g|nfc|chinh-hang.*|vn|vna)\b", "", s)
    return s.replace("-", " ").strip()


def build_targets(delay=2.0):
    """Fetch the public product sitemaps and map each of our thegioididong models to its CellphoneS product page."""
    import collections
    import sys
    import time
    import requests
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline_real import base_name, load_reviews
    idx = requests.get(SITEMAP, timeout=30, headers={"User-Agent": UA}).text
    urls = []
    for sm in re.findall(r"<loc>([^<]+product-sitemap[^<]*)</loc>", idx):
        urls += re.findall(r"<loc>([^<]+)</loc>", requests.get(sm, timeout=30, headers={"User-Agent": UA}).text)
        time.sleep(delay)
    (ROOT / "data" / "cps_sitemap_urls.txt").write_text("\n".join(sorted(set(urls))))
    models = collections.defaultdict(collections.Counter)
    for r in load_reviews():
        if r["source"] == "tgdd":
            models[str(r["modelId"])][r["slug"]] += 1
    names = {mid: _norm(base_name(min(c, key=len)).replace(" ", "-")) for mid, c in models.items()}
    by = collections.defaultdict(list)
    for u in urls:
        slug = u.rstrip("/").split("/")[-1]
        if not NOT_PHONE.search(slug):
            by[_norm(slug)].append(u)
    targets = {mid: min(by[n], key=len) for mid, n in names.items() if n in by}
    (ROOT / "data" / "cps_targets.json").write_text(json.dumps({"targets": targets, "names": names}, ensure_ascii=False, indent=1))
    print(f"{len(urls)} sitemap URLs; {len(targets)}/{len(names)} models matched")


def clean(m):
    """Keep review content only; drop customer identity."""
    return {"id": m.get("id"), "content": m.get("content"), "rating": m.get("rating_id"), "created_at": m.get("created_at"),
            "is_purchased": m.get("is_purchased"), "cps_product_id": m.get("product_id"),
            "attributes": [{"aspect": a.get("attribute_name"), "label": a.get("label")} for a in (m.get("attributes") or [])]}


def review_url(url):
    """https://cellphones.com.vn/<slug>.html → https://cellphones.com.vn/<slug>/review (the product's public review page)."""
    return re.sub(r"\.html$", "", url) + "/review"


async def crawl_product(page, url, max_pages, delay, buf):
    """Page 1 is server-rendered (read from the page's own state); later pages via the page's 'Xem thêm đánh giá' button."""
    got, pages = asyncio.Event(), {1}

    async def on_resp(r):
        q = r.request.post_data or ""
        if "graphql-customer" not in r.url or "reviews(" not in q:
            return
        try:
            data = ((await r.json()).get("data") or {}).get("reviews") or {}
        except Exception:
            return
        m = re.search(r"page\s*:\s*(\d+)", q)
        pages.add(int(m.group(1)) if m else 0)
        for x in data.get("matches") or []:
            buf.append(clean(x))
        got.set()

    page.on("response", on_resp)
    try:
        await page.goto(review_url(url), wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(delay * 1000)
        await page.keyboard.press("Escape")                     # close the province pop-up
        st = await page.evaluate("(() => { const r = (window.__NUXT__ || {}).state?.review || {};"
                                 " return {total: r.rating?.total_count ?? null, items: r.reviewData || []}; })()")
        for x in st["items"]:
            buf.append(clean(x))
        total = st["total"]
        while len(pages) < max_pages and total and len(buf) < total:
            btn = page.get_by_text("Xem thêm đánh giá", exact=True).first
            if await btn.count() == 0:
                break
            got.clear()
            try:
                await page.keyboard.press("Escape")
                await btn.scroll_into_view_if_needed()
                await btn.click(timeout=8000)
                await asyncio.wait_for(got.wait(), timeout=20)
            except Exception:
                break
            await page.wait_for_timeout(delay * 1000)
    finally:
        page.remove_listener("response", on_resp)
    return total, len(pages)


async def main_async(args):
    targets = json.loads((ROOT / "data" / "cps_targets.json").read_text())["targets"]
    state_p = ROOT / "data" / "cps_crawl_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    out_p = ROOT / "data" / "cps_reviews.jsonl"
    seen = {json.loads(l)["id"] for l in open(out_p)} if out_p.exists() else set()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(user_agent=UA, viewport={"width": 1280, "height": 900})
        with open(out_p, "a") as out:
            for n, (mid, url) in enumerate(targets.items(), 1):
                if state.get(mid, {}).get("done"):
                    continue
                buf = []
                try:
                    total, pages = await crawl_product(page, url, args.max_pages, args.delay, buf)
                    ok = True
                except Exception as e:
                    total, pages, ok = None, 0, False
                    print(f"  error {url}: {type(e).__name__}", flush=True)
                new = 0
                for r in buf:
                    if r["id"] in seen or not (r["content"] or "").strip():
                        continue
                    seen.add(r["id"])
                    out.write(json.dumps({"modelId": mid, "url": url, **r}, ensure_ascii=False) + "\n")
                    new += 1
                out.flush()
                state[mid] = {"url": url, "total": total, "pages": pages, "new": new, "done": ok}
                state_p.write_text(json.dumps(state, indent=1))
                print(f"[{n}/{len(targets)}] {url.split('/')[-1]} total={total} pages={pages} new={new}", flush=True)
                await asyncio.sleep(args.delay)
        await browser.close()
    print("reviews:", len(seen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pages", type=int, default=15)
    ap.add_argument("--delay", type=float, default=5.0)
    ap.add_argument("--targets", action="store_true", help="(re)build data/cps_targets.json from the sitemaps")
    args = ap.parse_args()
    if args.targets:
        return build_targets()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
