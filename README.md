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
| `src/edr.py` | **EDR, our model**: hedonic expectation + disconfirmation + γ chosen by held-out comparisons + Laplace posterior |
| `src/baselines.py` | baseline rankers: win-rate, Bradley–Terry, PageRank (+variants), PPR, random walker, pRank (Zhang 2010), fusion; metrics |
| `experiments/synthetic.py` | all synthetic experiments: E1 main table, E2 break-even, E3 sparsity, calibration → `results/synthetic/`, `figures/` |
| `src/absa_baseline.py`, `src/absa_transformer.py`, `src/absa_infer.py` | ABSA: TF-IDF baseline, PhoBERT multi-task training, CPU inference |
| `src/compare_rules.py`, `src/llm_extract.py` | comparative quintuple extraction: rules (L06) and Qwen2.5-7B few-shot (Kaggle) |
| `src/entity_link.py` | product-mention linking (catalogue / brand group / external phone) |
| `src/crawl_tgdd.py`, `src/crawl_prices.py`, `src/crawl_vnexpress.py` | polite crawlers (robots.txt, 5 s delay, no reviewer names); prices + spec sheets |
| `src/pipeline_real.py`, `src/real_rank.py` | real data: reviews → ABSA + comparisons → graph → all rankers + EDR, benchmark evaluation |
| `src/aspect_importance.py` | real-data finding: star ratings vs aspect polarity (loss aversion) |
| `src/run_suite.py`, `configs/`, `kaggle/push.sh` | run GPU experiment suites on Kaggle via the CLI |
| `Makefile` | `make synthetic`, `make real`, `make report`, `make all` |
| `results/`, `figures/` | metrics, study outputs, figures (checkpoints and crawled text are git-ignored) |

## Setup
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt && .venv/bin/python -m playwright install chromium
.venv/bin/python src/absa_baseline.py            # ~1 min, CPU
make synthetic                                   # all synthetic experiments + figures (~12 min CPU)
cd report && tectonic progress_report.tex        # build the report (XeLaTeX)
```
Crawled data and model checkpoints are not in git. Re-create them with the crawlers and `src/absa_transformer.py`.
