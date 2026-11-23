"""Crawl verified-buyer reviews per phone from thegioididong.com (the source site of UIT-ViSFD).

Loads each public review page (/dtdd/<slug>/danh-gia) in a headless browser and records the review data the
page itself fetches; further pages are reached by clicking the page's own pagination buttons.
Politeness: 5 s between page actions (robots.txt Crawl-delay: 5), one tab, resumable.
Privacy: reviewer names / customer ids are NOT stored — only text, star score, dates, product identifiers.

Input : data/tgdd_phone_urls.txt  (from the public product sitemap)
Output: data/tgdd_reviews.jsonl, data/tgdd_crawl_state.json
Usage : .venv/bin/python src/crawl_tgdd.py [--max_pages 30] [--delay 5]
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("commentId", "content", "score", "createdDate", "modelId", "variantCode", "productCode", "cbuyDate", "likeCount")


async def crawl_product(page, url, max_pages, delay, buf):
    """Returns (total, pages_seen)."""
    seen_pages, total = set(), None
    got = asyncio.Event()

    async def on_resp(r):
        nonlocal total
        if "Comment/Product/Code" not in r.url:
            return
        try:
            body = json.loads(r.request.post_data or "{}")
            data = (await r.json()).get("data") or {}
        except Exception:
            return
        total = data.get("total", total)
        seen_pages.add(body.get("pageIndex"))
        for c in data.get("comments") or []:
            buf.append({k: c.get(k) for k in KEEP})
        got.set()

    page.on("response", on_resp)
    try:
        await page.goto(url + "/danh-gia", wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(delay * 1000)
        nxt = 2
        while nxt <= max_pages and total and len(seen_pages) * 20 < total:
            btn = page.locator("button", has_text=str(nxt)).filter(has_not_text="»").first
            if await btn.count() == 0:
                btn = page.locator("button", has_text="»").first
                if await btn.count() == 0:
                    break
            got.clear()
            await btn.scroll_into_view_if_needed()
            await btn.click()
            try:
                await asyncio.wait_for(got.wait(), timeout=20)
            except asyncio.TimeoutError:
                break
            await page.wait_for_timeout(delay * 1000)
            nxt += 1
    finally:
        page.remove_listener("response", on_resp)
    return total, len(seen_pages)


async def main_async(args):
    urls = [u.strip() for u in open(ROOT / "data" / "tgdd_phone_urls.txt") if u.strip()]
    state_p = ROOT / "data" / "tgdd_crawl_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    out_p = ROOT / "data" / "tgdd_reviews.jsonl"
    seen_ids = {json.loads(l)["commentId"] for l in open(out_p)} if out_p.exists() else set()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(user_agent="Mozilla/5.0 (academic crawler; HUST IT4868 course project; 1 action / 5 s)")
        with open(out_p, "a") as out:
            for n, url in enumerate(urls, 1):
                slug = url.rstrip("/").split("/")[-1]
                if state.get(slug, {}).get("done"):
                    continue
                buf = []
                try:
                    total, pages = await crawl_product(page, url, args.max_pages, args.delay, buf)
                    ok = True
                except Exception as e:
                    total, pages, ok = None, 0, False
                    print(f"  error {slug}: {type(e).__name__}", flush=True)
                new = 0
                for r in buf:
                    if r["commentId"] in seen_ids:
                        continue
                    seen_ids.add(r["commentId"])
                    out.write(json.dumps({"slug": slug, **r}, ensure_ascii=False) + "\n")
                    new += 1
                out.flush()
                state[slug] = {"total": total, "pages": pages, "new": new, "done": ok}
                state_p.write_text(json.dumps(state, indent=1))
                print(f"[{n}/{len(urls)}] {slug} total={total} pages={pages} new={new}", flush=True)
                await asyncio.sleep(args.delay)
        await browser.close()
    print("reviews:", len(seen_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pages", type=int, default=30)
    ap.add_argument("--delay", type=float, default=5.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
