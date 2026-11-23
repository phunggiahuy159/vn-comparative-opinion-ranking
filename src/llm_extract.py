"""LLM-based comparative quintuple extraction (silver labels) with Qwen2.5-7B-Instruct.

Quintuple = (subject, object, aspect, predicate, label), VCOM/VLSP-2023 label set:
  COM+ / COM- / COM  : gradable comparison, subject better / worse / direction unclear
  SUP+ / SUP- / SUP  : superlative (best / worst / unclear)
  EQL                : equal
  DIF                : different, not gradable
Subject "THIS" = the product the review is posted under (implicit subject, L06-03 entity assignment).
Aspect is mapped to the UIT-ViSFD categories so it joins with the ABSA output.

Input : data/comparative_candidates.jsonl (buyer reviews, from compare_rules.py) or data/expert_candidates.jsonl (VnExpress)
Output: results/llm_tuples.jsonl (or --output)
Runs on Kaggle GPU (fp16 / 4-bit) or CPU (bf16, small model). Usage: python src/llm_extract.py [--input F --output F --limit N]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ASPECTS = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL", "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE"]
LABELS = ["COM+", "COM-", "COM", "SUP+", "SUP-", "SUP", "EQL", "DIF"]

SYSTEM = f"""You extract comparative opinions from Vietnamese smartphone reviews.
Return ONLY JSON: {{"comparisons": [{{"subject": str, "object": str|null, "aspect": str, "predicate": str, "label": str}}]}}
Rules:
- subject: the product being judged. If it is the reviewed phone and not named, use "THIS".
- object: the other product/brand compared against (copy the words from the text), or null for superlatives.
- aspect: one of {ASPECTS}.
- predicate: the comparative expression copied from the text (e.g. "trâu hơn", "không bằng", "nóng hơn").
- label: one of {LABELS}. COM+ = subject better, COM- = subject worse, COM = direction unclear,
  SUP+/SUP-/SUP = superlative best/worst/unclear, EQL = equal, DIF = different but not gradable.
  Negative predicates flip direction: "nóng hơn", "hao pin hơn" -> COM-.
- One entry per (object, aspect) pair. Comparisons with the phone's own price ("so với giá") or with
  older versions of itself count too. If there is no comparison between products, return {{"comparisons": []}}.
- Do not compare numbers that are not products (e.g. "48 chấm" is a camera spec, the product is the brand next to it)."""

SHOTS = [
    ("Pin trâu hơn con A50 nhưng camera không bằng Samsung S10, chơi game nóng hơn.",
     {"comparisons": [
         {"subject": "THIS", "object": "A50", "aspect": "BATTERY", "predicate": "trâu hơn", "label": "COM+"},
         {"subject": "THIS", "object": "Samsung S10", "aspect": "CAMERA", "predicate": "không bằng", "label": "COM-"},
         {"subject": "THIS", "object": "Samsung S10", "aspect": "PERFORMANCE", "predicate": "nóng hơn", "label": "COM-"}]}),
    ("Với s20 ultra mức giá này bằng với Oppo find x2 thì nên mua find x2 vì Oppo hơn hẳn con ultra về cấu hình, pin. Ultra chỉ hơn được về chụp ảnh.",
     {"comparisons": [
         {"subject": "Oppo find x2", "object": "s20 ultra", "aspect": "PRICE", "predicate": "bằng với", "label": "EQL"},
         {"subject": "Oppo find x2", "object": "s20 ultra", "aspect": "PERFORMANCE", "predicate": "hơn hẳn", "label": "COM+"},
         {"subject": "Oppo find x2", "object": "s20 ultra", "aspect": "BATTERY", "predicate": "hơn hẳn", "label": "COM+"},
         {"subject": "s20 ultra", "object": "Oppo find x2", "aspect": "CAMERA", "predicate": "hơn được về chụp ảnh", "label": "COM+"}]}),
    ("Máy đẹp, pin dùng được 1 ngày, nhân viên nhiệt tình nhất.",
     {"comparisons": []}),
]


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in SHOTS:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": text})
    return msgs


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for c in d.get("comparisons", []):
        if not isinstance(c, dict):
            continue
        if c.get("label") in LABELS and c.get("aspect") in ASPECTS:
            out.append({k: c.get(k) for k in ("subject", "object", "aspect", "predicate", "label")})
    return out


def load_model(name):
    """CPU: bfloat16 (use a small model, e.g. Qwen2.5-3B). GPU: fp16 if total memory >= 28 GB (2xT4); else 4-bit; else Qwen2.5-3B fp16."""
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, torch.get_num_threads()))
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16), f"{name}|cpu-bf16"
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30
    if total >= 28:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto"), f"{name}|fp16"
    try:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map="auto")
        # smoke test: one forward pass so unsupported kernels fail here, not mid-run
        with torch.no_grad():
            m(torch.tensor([[1, 2, 3]], device=m.device))
        return m, f"{name}|4bit"
    except Exception as e:
        print("4-bit load failed:", repr(e)[:300], flush=True)
        small = "Qwen/Qwen2.5-3B-Instruct"
        return AutoModelForCausalLM.from_pretrained(small, torch_dtype=torch.float16, device_map="auto"), f"{small}|fp16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--input", default="data/comparative_candidates.jsonl", help="jsonl rows with id, comment[, rule_tuples]")
    ap.add_argument("--output", default="results/llm_tuples.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / args.input)]
    if args.limit:
        rows = rows[:args.limit]
    model, used = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(used.split("|")[0], padding_side="left")
    print("model:", used, flush=True)
    model.eval()
    out_path = ROOT / args.output
    out_path.parent.mkdir(exist_ok=True)
    # sort by length so batches pad less
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["comment"]))
    t0, n_fail = time.time(), 0
    with open(out_path, "w") as f:
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            prompts = [tok.apply_chat_template(build_messages(rows[i]["comment"]), tokenize=False,
                                               add_generation_prompt=True) for i in idx]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, raw in zip(idx, texts):
                tuples = parse(raw)
                n_fail += tuples is None
                f.write(json.dumps({"model": used, "id": rows[i]["id"], "comment": rows[i]["comment"], "raw": raw,
                                    "llm_tuples": tuples or [], "parse_ok": tuples is not None,
                                    "rule_tuples": rows[i].get("rule_tuples", []),
                                    **{k: rows[i][k] for k in ("url", "title_product") if k in rows[i]}}, ensure_ascii=False) + "\n")
            f.flush()
            done = b + len(idx)
            print(f"{done}/{len(rows)}  {time.time() - t0:.0f}s  parse_fail={n_fail}", flush=True)
    print("done", out_path)


if __name__ == "__main__":
    main()
