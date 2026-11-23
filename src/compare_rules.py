"""Rule-based Vietnamese comparative opinion extraction (course L06-03 §5, adapted to Vietnamese).

L06-03 relation: <C (comparative word), F (feature/aspect), e1, e2, type>.
Vietnamese comparatives are analytic: "<pred> hơn <obj>", "không bằng <obj>", "thua <obj>", "ngang <obj>", "<pred> nhất".
Polarity rules (L06-03 "more/less + Pos/Neg"):
    pos-pred + hơn      -> subject better      ("pin trâu hơn A50")
    neg-pred + hơn      -> subject worse       ("nóng hơn", "hao pin hơn")
    kém/thua/không bằng -> subject worse       ("cấu hình không bằng Vsmart")
    bằng/ngang/như      -> equal
The subject is usually implicit (= the product being reviewed, L06-03 entity assignment), so we output
subject="THIS" unless an explicit product appears before the cue.

Outputs data/comparative_candidates.jsonl from UIT-ViSFD: every clause with a cue, with rule tuples.
Usage: .venv/bin/python src/compare_rules.py
"""
import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent

ASPECT_LEX = {
    "BATTERY": r"pin|sạc|dung lượng pin|hao pin|tụt pin|mah",
    "CAMERA": r"camera|cam\b|chụp|ảnh|quay|selfie|xóa phông|chấm|mp\b",
    "PERFORMANCE": r"cấu hình|chip|chíp|hiệu năng|mượt|lag|giật|đơ|game|chơi game|ram|xử lý|nóng|chạy|tốc độ|snapdragon|helio",
    "SCREEN": r"màn hình|màn\b|hiển thị|amoled|lcd|độ phân giải|cảm ứng",
    "DESIGN": r"thiết kế|đẹp|mỏng|nhẹ|cầm|vỏ|nắp lưng|kiểu dáng|màu",
    "PRICE": r"giá|tiền|rẻ|đắt|tầm giá|phân khúc",
    "FEATURES": r"vân tay|nhận diện|loa|wifi|wf|sóng|bluetooth|gps|face",
    "STORAGE": r"bộ nhớ|rom|gb\b",
}
POS_PRED = r"trâu|tốt|đẹp|ngon|mượt|nhanh|khỏe|khoẻ|mạnh|bền|nét|rõ|sắc nét|ổn|lâu|xịn|cao|sáng|to|ok|ổn định|nhạy|mát|rẻ|chất|hay|đã|tuyệt"
NEG_PRED = r"nóng|hao|tụt|yếu|chậm|lag|giật|đơ|tệ|dở|kém|mờ|tốn|đắt|nhanh hết|hụt|ít|thấp|tối|nhỏ|nặng|dày|xấu|rè|lỗi|hay hư"
BRANDS = r"iphone|ip\d+|samsung|ss\b|galaxy|oppo|xiaomi|redmi|mi\s?\d|vivo|realme|nokia|huawei|vsmart|asus|sony|lg\b|bphone|pixel|honor|poco|tecno|infinix|itel"
MODEL = rf"(?:(?:{BRANDS})(?:\s+[a-z]*\s?\d+[a-z]*(?:\s(?:pro|plus|ultra|max|lite|prime|s|e|neo|5g))*)?|(?:\b[ajmsfkx]\d{{1,2}}[a-z]?(?:\s(?:pro|plus|prime|s|e))?\b)|note\s?\d+(?:\s?pro)?|find\s?x\d|reno\s?\d+|y\d{{2}}s?\b|v\d{{2}}\b)"

CUES = [  # (regex, mode): mode "pred" = sign from predicate polarity, "win"/"lose" fixed, "equal", "sup"
    (rf"(?P<pred>(?:{NEG_PRED}|{POS_PRED})[\w\s]{{0,12}}?)\s+hơn\s+(?:cả\s+|hẳn\s+|nhiều\s+)?(?:con\s+|máy\s+|em\s+)?(?P<obj>{MODEL})", "pred"),
    (rf"(?P<pred>không bằng|ko bằng|k bằng|chẳng bằng|thua(?: xa)?(?: cả)?|kém(?: hơn)?(?: cả)?)\s+(?:con\s+|máy\s+|em\s+)?(?P<obj>{MODEL})", "lose"),
    (rf"(?P<pred>ăn đứt|vượt trội|vượt mặt|đè bẹp|hơn hẳn|hơn)\s+(?:con\s+|máy\s+)?(?P<obj>{MODEL})", "win"),
    (rf"(?P<pred>bằng(?: với)?|ngang(?: ngửa| với| bằng)?|tương đương(?: với)?|y như|giống)\s+(?:con\s+|máy\s+)?(?P<obj>{MODEL})", "equal"),
    (rf"(?P<pred>(?:{POS_PRED}|{NEG_PRED})[\w\s]{{0,8}}?)\s+nhất(?:\s+(?:trong\s+)?(?:tầm giá|phân khúc))?", "sup"),
]
ANY_CUE = re.compile(r"\b(hơn|bằng|thua|kém|nhất|so với|ngang|ăn đứt|vượt)\b")


def aspect_of(clause: str):
    for a, pat in ASPECT_LEX.items():
        if re.search(pat, clause):
            return a
    return "GENERAL"


def polarity(pred: str) -> int:
    neg = re.search(rf"\b(?:{NEG_PRED})", pred) is not None
    pos = re.search(rf"\b(?:{POS_PRED})", pred) is not None
    flip = re.search(r"\b(không|ko|k|chẳng|chưa)\b", pred) is not None
    s = -1 if neg and not pos else (1 if pos and not neg else 1)
    return -s if flip else s


def extract(text: str):
    t = re.sub(r"\s+", " ", text.lower())
    out = []
    for clause in re.split(r"[.!?;\n]|,(?=\s*(?:nhưng|còn|mà|tuy))| nhưng | còn ", t):
        clause = clause.strip()
        if not ANY_CUE.search(clause):
            continue
        for pat, mode in CUES:
            m = re.search(pat, clause)
            if not m:
                continue
            pred = m.group("pred").strip()
            obj = m.groupdict().get("obj")
            pre = clause[:m.start()]  # explicit subject = last product mentioned before the cue
            subj_m = list(re.finditer(MODEL, pre))
            subj = subj_m[-1].group(0) if subj_m else "THIS"
            if mode == "equal":
                label, typ = "EQL", "equal"
            elif mode == "sup":
                label, typ = ("SUP+" if polarity(pred) > 0 else "SUP-"), "superlative"
            else:
                sign = {"win": 1, "lose": -1}.get(mode) or polarity(pred)
                label, typ = ("COM+" if sign > 0 else "COM-"), "better"
            out.append({"clause": clause, "subject": subj, "object": obj.strip() if obj else None,
                        "aspect": aspect_of(clause), "predicate": pred, "label": label, "rule_type": typ})
            break
    return out


def main():
    df = load_dataset("visolex/ViSFD")["train"].to_pandas()
    out_path = ROOT / "data" / "comparative_candidates.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    n_cue = n_tuple = 0
    labels = {}
    with open(out_path, "w") as f:
        for r in df.itertuples():
            if not ANY_CUE.search(r.comment.lower()):
                continue
            n_cue += 1
            tuples = extract(r.comment)
            n_tuple += bool(tuples)
            for t in tuples:
                labels[t["label"]] = labels.get(t["label"], 0) + 1
            f.write(json.dumps({"id": int(r.index), "split": r.type, "comment": r.comment, "n_star": int(r.n_star),
                                "absa_label": r.label, "rule_tuples": tuples}, ensure_ascii=False) + "\n")
    stats = {"comments": len(df), "with_cue": n_cue, "with_rule_tuple": n_tuple, "label_counts": labels}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "compare_rules_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
