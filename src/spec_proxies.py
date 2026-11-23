"""Objective spec-sheet proxies used as ground truth when lab benchmarks are unavailable.

From each product page's schema.org spec sheet (data/tgdd_products_specs.jsonl):
  BATTERY      ← battery capacity (mAh)            Android only (iPhones list video-playback hours, not mAh)
  SCREEN       ← peak brightness (nits)            + refresh rate, OLED flag as covariates
  CAMERA       ← main camera MP                    weak proxy (MP ≠ quality) — reported but flagged
  PERFORMANCE  ← chipset → AnTuTu (team fills data/chip_antutu.csv; template is generated)
Feature phones are excluded (no Android/iOS).
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMART_OS = re.compile(r"android|ios|hyperos|magicos|coloros|one ui|funtouch|realme ui|harmony", re.I)


def _fix(s):
    """Repair UTF-8 text that was decoded as Latin-1 (mojibake), e.g. 'Há»\x87' → 'Hệ'."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _num(pattern, text, cast=float):
    m = re.search(pattern, text or "", re.I)
    return cast(m.group(1).replace(".", "").replace(",", "")) if m else None


def parse(specs):
    s = {_fix(k): _fix(v) for k, v in (specs or {}).items()}
    os_ = s.get("Hệ điều hành", "")
    out = {"smartphone": bool(SMART_OS.search(os_)), "os": os_, "chip": s.get("Chip xử lý (CPU)"),
           "launch": s.get("Thời gian ra mắt")}
    batt = s.get("Dung lượng pin", "")
    out["battery_mah"] = _num(r"(\d[\d.,]*)\s*mAh", batt)
    out["screen_nits"] = _num(r"(\d[\d.,]*)\s*nits", s.get("Độ sáng", ""))
    out["refresh_hz"] = _num(r"(\d+)\s*Hz", s.get("Tần số quét", ""))
    out["oled"] = int(bool(re.search(r"OLED", s.get("Công nghệ màn hình", ""), re.I)))
    out["main_camera_mp"] = _num(r"Chính\s*(\d+(?:\.\d+)?)\s*MP", s.get("Độ phân giải camera sau", "")) or \
        _num(r"(\d+(?:\.\d+)?)\s*MP", s.get("Độ phân giải camera sau", ""))
    out["ram_gb"] = _num(r"(\d+)\s*GB", s.get("RAM", ""))
    return out


def load(products):
    """products: {modelId: {"slugs": [...]}} → {modelId: proxy dict} (first slug that has a spec sheet)."""
    p = ROOT / "data" / "tgdd_products_specs.jsonl"
    by_slug = {}
    if p.exists():
        for l in open(p):
            d = json.loads(l)
            if d.get("specs"):
                by_slug.setdefault(d["slug"], parse(d["specs"]))
    out = {}
    for k, v in products.items():
        for sl in v.get("slugs", []):
            if sl in by_slug:
                out[k] = by_slug[sl]; break
    return out


def chip_scores():
    """Team-filled data/chip_antutu.csv (chip,antutu) → {chip: score}; writes a template listing chips if missing."""
    p = ROOT / "data" / "chip_antutu.csv"
    if p.exists():
        return {r["chip"]: float(r["antutu"]) for r in csv.DictReader(open(p)) if r.get("antutu")}
    return {}


def write_chip_template(proxies):
    chips = sorted({v["chip"] for v in proxies.values() if v.get("chip") and v.get("smartphone")})
    p = ROOT / "data" / "chip_antutu_template.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["chip", "antutu", "source_url"])
        for c in chips:
            w.writerow([c, "", ""])
    return p, len(chips)


def targets(proxies, chip=None):
    """Aspect → {modelId: ground-truth proxy value} for smartphones."""
    chip = chip or {}
    t = {"BATTERY": {}, "SCREEN": {}, "CAMERA": {}, "PERFORMANCE": {}}
    for k, v in proxies.items():
        if not v["smartphone"]:
            continue
        if v["battery_mah"] and "ios" not in v["os"].lower():
            t["BATTERY"][k] = v["battery_mah"]
        if v["screen_nits"]:
            t["SCREEN"][k] = v["screen_nits"]
        if v["main_camera_mp"]:
            t["CAMERA"][k] = v["main_camera_mp"]
        if v.get("chip") in chip:
            t["PERFORMANCE"][k] = chip[v["chip"]]
    return t
