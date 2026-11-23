"""Link free-text product mentions in Vietnamese reviews to the crawled catalogue (course L06-03 / L07).

Mention → one of:
  * a catalogue product key (modelId) — e.g. "ip 15 pro" → iphone-15-pro
  * "group:<brand>"   — brand-only mentions ("hơn samsung", "không bằng ss")
  * "ext:<mention>"   — a real phone outside our catalogue (e.g. older "j7 pro"); kept as a graph node, it still
                        links catalogue phones that were compared against it, but it has no benchmark.
"""
import re
from collections import defaultdict

BRAND_ALIAS = {"ss": "samsung", "sam": "samsung", "samsum": "samsung", "galaxy": "samsung", "ip": "iphone", "iph": "iphone",
               "apple": "iphone", "táo": "iphone", "redmi": "xiaomi", "poco": "xiaomi", "mi": "xiaomi", "xiaomi": "xiaomi",
               "oppo": "oppo", "vivo": "vivo", "realme": "realme", "honor": "honor", "nokia": "nokia", "huawei": "huawei",
               "vsmart": "vsmart", "motorola": "motorola", "moto": "motorola", "tecno": "tecno", "infinix": "infinix",
               "itel": "itel", "nubia": "nubia", "nothing": "nothing", "asus": "asus", "sony": "sony", "pixel": "google",
               "iphone": "iphone", "samsung": "samsung"}
FILLER = r"\b(con|máy|em|chiếc|điện thoại|đt|dt|cái|bản|dòng|hàng|của|bên|thằng|mấy)\b"
STOP = {"5g", "4g", "chính", "hãng", "điện", "thoại"}


def norm_tokens(text):
    t = text.lower()
    t = re.sub(FILLER, " ", t)
    t = re.sub(r"\bpro\s?max\b|\bprm\b|\bpm\b", " pro max ", t)
    t = re.sub(r"\b\d+(tr|triệu|k|mah|mp|chấm|gb|hz|w)\b", " ", t)      # prices / specs are not product names
    t = re.sub(r"\b(ip|iphone)\s?(\d{1,2})", r"iphone \2", t)            # ip15 → iphone 15
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t) if re.search(r"\b(iphone|note|reno|find)\d", t) else t
    toks = [BRAND_ALIAS.get(w, w) for w in re.findall(r"[a-z0-9]+", t)]
    return [w for w in toks if w not in STOP]


class Catalogue:
    def __init__(self, products):
        """products: dict key -> {"name": str (from slug), "brand": str, "n_reviews": int}"""
        self.p = products
        self.tokens = {k: set(norm_tokens(v["name"])) for k, v in products.items()}
        self.brands = {v["brand"] for v in products.values()}
        self.by_brand = defaultdict(list)
        for k, v in products.items():
            self.by_brand[v["brand"]].append(k)

    def link(self, mention):
        toks = norm_tokens(mention or "")
        if not toks:
            return None
        brand = next((t for t in toks if t in self.brands or t in BRAND_ALIAS.values()), None)
        rest = [t for t in toks if t != brand and t not in {"samsung", "galaxy"}]
        if not rest:
            return f"group:{brand}" if brand else None
        cands = []
        for k, tk in self.tokens.items():
            if brand and self.p[k]["brand"] != brand:
                continue
            if all(t in tk for t in rest):
                extra = len(tk - set(rest) - {self.p[k]["brand"], "galaxy"})  # prefer the base model over Pro/Max
                cands.append((extra, -self.p[k].get("n_reviews", 0), k))
        if cands:
            return sorted(cands)[0][2]
        return "ext:" + " ".join(([brand] if brand else []) + rest)
