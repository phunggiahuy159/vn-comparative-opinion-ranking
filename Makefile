# One-command reproduction. CPU-only targets run locally; GPU training goes through Kaggle (see kaggle/push.sh).
PY := .venv/bin/python

.PHONY: all env absa-baseline synthetic findings data real report clean-figures

all: absa-baseline synthetic findings real report

env:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

absa-baseline:            ## TF-IDF + LR on UIT-ViSFD (~1 min)
	$(PY) src/absa_baseline.py

synthetic:                ## E1 main table, E2 break-even, E3 sparsity → results/synthetic, figures/ (~12 min CPU)
	$(PY) experiments/synthetic.py

findings:                 ## real-data finding: aspect importance / loss aversion on ViSFD
	$(PY) src/aspect_importance.py

data:                     ## polite crawls (robots.txt, 5 s delay) — slow; resumable
	$(PY) -u src/crawl_tgdd.py
	$(PY) -u src/crawl_prices.py data/tgdd_phone_urls.txt
	$(PY) -u src/crawl_vnexpress.py
	$(PY) src/crawl_cellphones.py --targets && $(PY) -u src/crawl_cellphones.py

real:                     ## real pipeline (needs data/ + PhoBERT checkpoint) → graph → EDR rankings
	$(PY) src/pipeline_real.py
	$(PY) src/real_rank.py

report:                   ## tables from results/ (report/make_tables.py), then both LaTeX reports (ICLR 2027 template, tectonic)
	$(PY) report/make_tables.py
	cd report && tectonic progress_report.tex && cp progress_report.pdf Group20_ProgressReport.pdf
	cd report && tectonic final_report.tex