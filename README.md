# Vietnamese Aspect-Based Opinion Mining and Comparative Product Ranking

IT4868E Web Mining (HUST), capstone project, Group 20:
Phùng Gia Huy (20235506), Vũ Hải An (20235468), Chu Xuân Minh (20235527), Nguyễn Bình Anh (20235470).

We mine Vietnamese smartphone reviews for aspect sentiment and comparative opinions, rank phones per aspect
(battery, camera, performance, screen) and validate the rankings against objective benchmarks. Our model **EDR
(Expectation–Disconfirmation Ranking)** jointly explains comparisons (Bradley–Terry) and expectation-adjusted
sentiment. It corrects the popularity and price-relative-sentiment biases of PageRank-style rankers such as pRank
(Zhang et al., 2010).

- **Current status, open items and next steps: [STATUS.md](STATUS.md)** ← start here
- Full plan, design decisions, related work: [PLAN.md](PLAN.md)
- Progress report (LaTeX): [report/progress_report.tex](report/progress_report.tex) → `report/Group20_ProgressReport.pdf`

## Layout
| Path | What |
|---|---|
| `src/absa_baseline.py` | TF-IDF + LR ABSA baseline on UIT-ViSFD |
| `src/absa_transformer.py` | PhoBERT multi-task ABSA (one 4-way head per aspect); MPS/CUDA/CPU |
| `src/absa_infer.py` | apply a trained ABSA checkpoint to new text (CPU) |
| `src/compare_rules.py` | rule-based Vietnamese comparative extraction (L06), quintuples |
| `src/llm_extract.py` | Qwen2.5-7B few-shot comparative quintuple extraction (silver labels; Kaggle) |
| `src/entity_link.py` | product-mention linking (catalogue / brand group / external phone) |
| `src/ranking.py` | rankers: win-rate, Bradley–Terry, PageRank (+variants), PPR, random walker, pRank, fusion; synthetic study 1 |
| `src/joint_model.py` | **EDR model** (MAP, noise-aware BT, set comparisons, expectation covariates, bootstrap ranks) |
| `src/study_advanced.py` | synthetic study 2: realistic failure modes + break-even curve |
| `src/aspect_importance.py` | stars ~ aspect polarity regression (loss-aversion finding) |
| `src/crawl_tgdd.py`, `src/crawl_prices.py`, `src/crawl_vnexpress.py` | polite crawlers (robots.txt, 5 s delay, no reviewer names) |
| `src/pipeline_real.py`, `src/real_rank.py` | real-data pipeline: reviews → graph → rankings → benchmark eval |
| `src/run_suite.py`, `configs/`, `kaggle/push.sh` | run experiment suites on Kaggle GPUs via the Kaggle CLI |
| `results/` | metrics and study outputs (checkpoints are git-ignored) |

## Setup
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt torch transformers pyvi playwright requests pypdf
.venv/bin/python src/absa_baseline.py            # ~1 min, CPU
.venv/bin/python src/ranking.py                  # synthetic study 1
.venv/bin/python src/study_advanced.py           # synthetic study 2 (~2 min)
cd report && tectonic progress_report.tex        # build the report (XeLaTeX)
```
Crawled data and model checkpoints are not in git. Re-create them with the crawlers and `src/absa_transformer.py`.
