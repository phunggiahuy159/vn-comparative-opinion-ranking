"""Run a list of experiment configs across all visible GPUs (one queue worker per GPU), then aggregate.

Usage: python src/run_suite.py configs/absa_suite.json
Each config: {"script": "src/absa_transformer.py", "args": "--model ... --seed 42", "tag": "..."}
Writes results/logs/<tag>.log and results/suite_summary.csv (mean ± std over seeds per model).
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def n_gpus():
    try:
        import torch
        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def worker(gpu, q, status):
    while True:
        try:
            cfg = q.get_nowait()
        except queue.Empty:
            return
        log = ROOT / "results" / "logs" / f"{cfg['tag']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, "-u", str(ROOT / cfg["script"])] + cfg["args"].split() + ["--tag", cfg["tag"]]
        t0 = time.time()
        print(f"[gpu{gpu}] START {cfg['tag']}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
        status[cfg["tag"]] = {"rc": rc, "min": round((time.time() - t0) / 60, 1)}
        print(f"[gpu{gpu}] END {cfg['tag']} rc={rc} {status[cfg['tag']]['min']}min", flush=True)
        if rc != 0:
            print("".join(open(log).readlines()[-15:]), flush=True)


def aggregate():
    rows = []
    for m in sorted((ROOT / "results").glob("absa_*/metrics.json")):
        d = json.loads(m.read_text())
        rows.append({"tag": m.parent.name.removeprefix("absa_"), "model": d["model"], "seed": d["args"]["seed"],
                     "det_microF1": d["aspect_detection_microF1"], "det_macroF1": d["aspect_detection_macroF1"],
                     "pair_microF1": d["aspect_polarity_microF1"],
                     **{f"{a}_pair": d["per_aspect"][a]["pair_microF1"] for a in ("BATTERY", "CAMERA", "PERFORMANCE", "SCREEN")}})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "suite_runs.csv", index=False)
    num = df.drop(columns=["tag", "seed"]).groupby("model")
    summ = num.mean().round(4).add_suffix("_mean").join(num.std().round(4).add_suffix("_std")).join(num.size().rename("n_seeds"))
    summ.to_csv(ROOT / "results" / "suite_summary.csv")
    print(summ[["n_seeds", "det_microF1_mean", "det_microF1_std", "pair_microF1_mean", "pair_microF1_std"]].to_string())


def main():
    cfgs = json.loads(Path(sys.argv[1]).read_text())
    q = queue.Queue()
    for c in cfgs:
        q.put(c)
    status, g = {}, n_gpus()
    print(f"{len(cfgs)} configs on {g} GPU(s)", flush=True)
    ts = [threading.Thread(target=worker, args=(i, q, status)) for i in range(g)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    (ROOT / "results" / "suite_status.json").write_text(json.dumps(status, indent=2))
    aggregate()


if __name__ == "__main__":
    main()
