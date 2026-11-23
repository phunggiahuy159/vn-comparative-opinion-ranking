# STATUS / HANDOFF — read this first

Last updated: **23/11/2026, ~21:00 (UTC+7)**. This is the single source of truth for "where are we, what is blocked,
what's next". Details and rationale are in [PLAN.md](PLAN.md). The proposal is fixed: the instructor requires consistency between
proposal, progress report and final report.

## 1. One-paragraph summary
Mine Vietnamese phone reviews → aspect sentiment (PhoBERT) + comparative opinions (rules / LLM) → entity linking → rank phones
per aspect → validate against benchmarks (GSMArena / DXOMARK / AnTuTu). The comparative-graph + PageRank idea is **prior art**
(Zhang et al. 2010, "pRank"). We reproduce it as a baseline. Our contributions:
1. the Vietnamese pipeline;
2. evidence that PageRank-style rankers measure popularity, and that pRank ≈ sentiment ranking that collapses when sentiment is price-relative;
3. **EDR, Expectation–Disconfirmation Ranking** ([src/joint_model.py](src/joint_model.py)): a joint Bradley–Terry + expectation-adjusted
   sentiment model grounded in Oliver (1980);
4. a break-even analysis (EDR stays above price-only ranking up to extraction direction error ε = 0.4, margin .17 → .04; BT + sentiment falls below price at ε ≈ 0.1);
5. real-data findings (loss aversion in star ratings; buyers compare against their old phone or a brand, not against competitors).

## 2. Done (with evidence)
| Item | Result | Where |
|---|---|---|
| ABSA baseline (TF-IDF + LR), ViSFD test | aspect det. micro-F1 0.829, aspect#polarity 0.710 | `results/absa_baseline_*` |
| **PhoBERT-base-v2 multi-task ABSA** (trained once locally on MPS, 6 epochs) | det. 0.890, **aspect#pol. 0.815**; BATTERY .856, CAMERA .870, PERF .829, SCREEN .771 | `results/absa_phobert-base-v2/metrics.json` (checkpoint is local only, git-ignored) |
| Rule comparative extractor on ViSFD | 3,051 comments with a cue → tuples for 510 | `results/compare_rules_stats.json` |
| Aspect importance (real, ViSFD n = 11,122) | R² .615; negatives cost 1.5–2.7× more stars than positives add (DESIGN is the exception) | `results/aspect_importance.txt` |
| thegioididong crawl (2023–26 phones) | 279 pages → ~1.5k reviews, ~95 models, only ~22 with ≥ 20 reviews | `data/` (git-ignored) |
| Real pipeline end to end (CPU) | 21 linked comparisons: 9 vs the reviewer's *old* phone, 8 vs a *brand*, 1 head-to-head | `results/real_pipeline_stats.json` |
| Progress report (LaTeX, 3 pages, Group 20) | compiled; ICLR format; describes the current EDR (hedonic prior, γ by held-out CV), 20-seed E1 table, break-even figure; no week labels | `report/progress_report.tex` → `report/Group20_ProgressReport.pdf` |
| Final report draft | ICLR format; **[TODO: …]** marks the real-data results still missing | `report/final_report.tex` |
| Report format | official ICLR 2027 template (black-and-white layout, colour figures, dark-blue links); Times via TeX Gyre Termes for Vietnamese | `report/iclr2027_conference.sty` |
| **EDR final** (hedonic prior + disconfirmation + γ chosen by held-out comparisons + Laplace) | E1 (20 seeds, ε = .1): EDR .855 / .833 / .818 at γ = 0 / .5 / .8 vs pRank .841 / .778 / .664, price .634; ablations all lower; ECE .080. E2 break-even, E3 sparsity, calibration figures | `src/edr.py`, `experiments/synthetic.py` → `results/synthetic/summary.txt`, `figures/` |
| Real-data EDR run (prices for 99/102 phones) | 90% rank intervals are wide for most phones (e.g. (1,3,16)), tight only for well-reviewed ones (Moto G86 battery 4–9; iPhone 17 Pro Max perf 1–10). Hedonic log-price slope: battery −0.08 (budget phones have big batteries), camera +0.11, perf +0.10, screen +0.08. γ is WEAK (0 entity-level comparisons) | `src/real_rank.py`, `results/real_eval.json` |
| Expert-comparison path (VnExpress, title = implicit subject) | code ready; rules give few, noisy comparisons → needs the LLM | `src/pipeline_real.py::comparisons_articles` |

## 3. Running / waiting (as of the timestamp above)
| What | State | How to check / resume |
|---|---|---|
| Kaggle CLI login | renewed 23/11 ~21:10 (`kaggle auth login --force` when the API says 'Authentication required') | |
| Kaggle `phunggiahuy/vabsa-absa-suite` (9 ABSA runs: PhoBERT-large ×3, base ×3, XLM-R, no-seg ablation) | **QUEUED for hours** (Kaggle T4 shortage; CPU jobs run fine) | `kaggle kernels status phunggiahuy/vabsa-absa-suite`; fetch with `kaggle kernels output phunggiahuy/vabsa-absa-suite -p kaggle/out/suite` |
| Kaggle `phunggiahuy/vabsa-llm-extract` (Qwen2.5-7B silver quintuples for 3,051 ViSFD candidates) | **QUEUED** | same commands |
| VnExpress expert-article crawl | running (~500 phone article links found) | `logs/crawl_vnexpress.log`; resumable: `.venv/bin/python -u src/crawl_vnexpress.py` |
| Price crawl (JSON-LD on product pages) | done (99/102 review-bearing models priced) | `logs/crawl_prices.log`; resumable: `.venv/bin/python -u src/crawl_prices.py data/tgdd_phone_urls.txt` |

## 4. Waiting on humans (decisions / team tasks)
1. **GPU route** for the LLM extractor and the large models, while Kaggle is stuck: (a) wait, (b) allow the owner's Mac GPU for the LLM job only
   (the owner asked NOT to use the local GPU by default), or (c) Modal account.
2. **More review sources?** FPT Shop / CellphoneS: needs the owner's OK. Review counts per phone are too low for robust real-data evaluation.
3. **Benchmarks spreadsheet**: fill `data/benchmarks_template.csv` (committed; regenerated by `src/real_rank.py`):
   GSMArena battery hours, DXOMARK camera/display, AnTuTu → save as `data/benchmarks.csv`.
4. **Request VCOM** (VLSP 2023 ComOM) from the organisers. It is the comparative-extraction benchmark and comes from VnExpress/Tinhte text.
5. Revoke the GitHub token that was pasted into chat on 23/11 and use `gh auth login` instead.
6. The repo https://github.com/phunggiahuy159/vn-comparative-opinion-ranking is **public** (since 23/11); the report links to it. Keep scraped data, checkpoints and slides out of git (see .gitignore).

## 5. Next steps (in order)
0. (done) `src/real_rank.py` uses the final EDR; it falls back to learned γ (flagged WEAK) when there are < 15 entity-level comparisons.
1. When prices finish: `python src/pipeline_real.py && python src/real_rank.py` (EDR then learns β_price per aspect).
2. Run comparative extraction on the VnExpress sentences (rules now, LLM when a GPU is available), link them to the catalogue, and add
   them as **expert comparisons** to the graph. EDR fuses buyer sentiment (thegioididong) with expert comparisons (VnExpress).
3. When the LLM labels arrive: measure agreement with the rules, then annotate a 300–500-sentence gold set → **measure ε** → place it on the break-even curve.
4. Distil the LLM labels into a PhoBERT extractor (Kaggle).
5. When benchmarks are in: Spearman / Kendall / NDCG + bootstrap CIs per aspect, overall and within price tiers; held-out comparison accuracy.
6. Final report + slides + small demo (aspect → ranked phones + rank intervals + review evidence per edge).

## 6. Constraints and conventions for any session
- **Do not use the local GPU** (owner's request). Use Kaggle (`kaggle/push.sh <slug> "<cmd>" "<pip>" files…`, account `phunggiahuy`, max 2 GPU batch sessions) or CPU.
- Crawling: respect robots.txt and 5 s delays; store no reviewer names. thegioididong reviews are collected by **rendering the public review page**
  (Playwright) — directly calling its API with spoofed browser headers was blocked by the permission system; do not retry that.
  **Tinhte disallows ClaudeBot**, so do not crawl it.
- Report honestly: synthetic studies show mechanisms only. Real accuracy claims need the benchmark correlations.
- Commits: author `phunggiahuy159 <phunggiahuy.15092005@gmail.com>`, **no co-author trailer** (owner's request).
- Python env: `uv venv --python 3.11 .venv`. The LaTeX report builds with `tectonic` (XeLaTeX).
