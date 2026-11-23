# Vietnamese Aspect-Based Opinion Mining & Comparative Product Ranking — Project Plan

IT4868E Web Mining (HUST), capstone. Status as of 23/11/2026. Keep this consistent with the proposal:
*mine Vietnamese phone reviews → aspects + aspect sentiment + comparative opinions → per-aspect comparison
graph → PageRank ranking → validate against real benchmark scores.*

---

## 1. How the project maps onto the course

| Pipeline stage | Course lecture | What we use from it |
|---|---|---|
| Overall process (crawl → clean → mine → evaluate → present) | L01 Intro (KDD process) | Report structure |
| Crawling, Vietnamese preprocessing (word segmentation, teencode normalisation) | L04 IR §5 text preprocessing | `pyvi`/VnCoreNLP segmentation, dedup |
| Aspect category detection + aspect polarity | L06-02 aspect extraction & aspect-level sentiment; L02 NB/SVM/LR, CNN/RNN | TF-IDF+LR baseline → PhoBERT/ViSoBERT |
| Product-name detection, linking "con này / máy này" to the reviewed phone | L06-03 [4] entity detection & assignment; L07 NER, coreference | Implicit subject = product page; alias dictionary + fuzzy match |
| Comparative sentence mining ⟨C, F, e1, e2, type⟩ | L06-03 [5] comparative sentences (better / equal / superlative / non-gradable; -er rules, "less + Pos → Neg") | Rule baseline in Vietnamese (`hơn`, `kém`, `thua`, `bằng`, `nhất`, `không bằng`…) → learned extractor |
| Comparison graph & ranking | L05 link analysis (PageRank, HITS) | Weighted PageRank, personalized PageRank, HITS |
| Fake/duplicate reviews | L06-03 opinion spam, L04 web spam | Near-duplicate filtering, rating/text mismatch |
| Evaluation | L02 evaluation methods, L04 IR evaluation | P/R/F1; Spearman, Kendall τ, NDCG@k vs benchmarks |
| Visualising graphs & rankings | L03 Data visualization | Graph plots, bump charts per aspect |

---

## 2. Related work and our contribution

**Closest prior work: Zhang, Narayanan & Choudhary (WOSN 2010), "Voice of the Customers: Mining Online Customer
Reviews for Product Feature-based Ranking".** They build one directed, weighted product graph per feature from
comparative sentences (edge weight = #positive / #negative comparatives). Their node weight is the #positive /
#negative ratio of subjective sentences. They rank with **pRank**, a PageRank variant multiplied by the node weight,
on English Amazon reviews of cameras and TVs, and evaluate against subject experts, including per price bin.
→ *The idea "comparative graph + PageRank" is not new. We reproduce pRank as a baseline
([src/ranking.py](src/ranking.py) `prank`) and do not claim it as our contribution.*

**Our contribution, relative to Zhang et al.:**
1. **Vietnamese comparative mining.** Vietnamese comparatives are analytic (`hơn`, `không bằng`, `thua`), predicates
   can reverse the direction (`nóng hơn` → worse), and the subject is usually implicit (`con này`). We compare
   rules (L06-03), PhoBERT and LLM-silver extraction under the VLSP-2023 VCOM label scheme.
2. **Objective validation.** We correlate rankings with DXOMARK / GSMArena / AnTuTu scores, with bootstrap
   confidence intervals, instead of comparing with expert opinion.
3. **Analysis of the rankers themselves.** In our synthetic study, pRank ≈ sentiment-only ranking at every
   sentiment-bias level. Its multiplicative node weight swamps the comparison graph, and its graph-only part is
   popularity-biased. When direct sentiment is price-relative, which is typical of Vietnamese reviews
   ("ngon trong tầm giá"), pRank collapses. A statistical paired-comparison model fused with sentiment stays
   robust. We test whether this holds on real data.
4. **(Optional) Robustness to extraction noise.** How much does each ranker degrade under the extractor's measured error rates?

## 3. Design decisions (the "advanced" part)

**D1. Plain PageRank measures popularity, not quality, so we keep it but correct it.**
A sanity check on synthetic data ([src/ranking.py](src/ranking.py), 30 products, 400 noisy comparisons,
Zipf-like popularity) shows plain PageRank on loser→winner edges reaches only
Spearman ≈ 0.25 against the hidden true quality. Bradley–Terry gets ≈ 0.65 and win-rate ≈ 0.68 (30 seeds; full table in §6).
The reason: a phone that is *mentioned* a lot collects rank even with a mediocre win ratio.
The fixes we test:
- degree-normalised PageRank (≈ 0.41)
- **personalized PageRank (PPR)**, which uses the ABSA sentiment prior as its teleport vector (≈ 0.40; ≈ 0.55 once degree-normalised)
- Callaghan–Mucha–Porter **random-walker ranking** (≈ 0.56)

Synthetic numbers only show the *direction* of the effect. The generator is Bradley–Terry, so it favours BT by construction. Real conclusions must come from benchmark correlation.
→ Research question RQ2 for the final report: *does graph structure add information beyond counting
wins and sentiment?*

**D2. Fuse the 90% with the 10%.** Only about 8–10% of reviews compare two named phones. In ViSFD, 23.5% of comments contain a
comparative cue and 7.7% contain a cue plus a phone name, which matches the ~10% figure in L06-03.
Pure comparison graphs are therefore sparse. PPR lets the direct aspect sentiment (the other 90%) act as
the prior while comparisons redistribute mass. This is the core methodological contribution.

**D3. Implicit subject.** Most comparisons look like *"pin trâu hơn con A50"*: the subject is the phone whose
page the review sits on. Product-linked crawling solves subject assignment for free. The L06-03 rules
(sentiment consistency between sentences) handle the remaining cases.

**D4. Edge direction & weights.** A *better* comparison adds an edge loser→winner, weighted by extractor confidence.
An *equal* comparison adds half-weight edges both ways. Superlatives (`nhất`) are skipped, or count as a
win against the phone's own price segment. Negative predicates flip the direction (`nóng hơn`, `hao pin hơn` →
the subject *loses*), following L06-03's "C × F polarity" rules.

**D5. Price confound.** Vietnamese reviewers judge *within a price range* ("ngon trong tầm giá"), while
benchmarks are absolute. We report correlation both overall and **within price tiers** (<5M, 5–10M, >10M VND), plus a partial Spearman that controls for launch price. Without this, a "low correlation" result is uninterpretable.

**D6. Aspect → benchmark mapping.** We evaluate only aspects that have an objective benchmark:

| Aspect (ViSFD label) | Benchmark | Notes |
|---|---|---|
| BATTERY | GSMArena battery test (Active Use Score) and/or DXOMARK Battery | Use one test version consistently; GSMArena covers more mid-range models |
| CAMERA | DXOMARK Camera | Mostly flagships, so coverage risk for budget phones |
| PERFORMANCE | AnTuTu / Geekbench 6 multi-core | Wide coverage |
| SCREEN | DXOMARK Display | Optional |

GENERAL, PRICE, DESIGN and SER&ACC are ranked but not benchmark-evaluated. They are reported qualitatively.

---

## 4. Data

| ID | Dataset | Size | Role | Status |
|---|---|---|---|---|
| D1 | **UIT-ViSFD** (Phan et al. 2021), HF `visolex/ViSFD` | 11,122 comments (7,786 / 1,112 / 2,224); 10 aspects + OTHERS, 3 polarities | Train/evaluate ABSA | ✅ downloaded, baseline done |
| D2 | **VCOM** (VLSP 2023 ComOM) | 120 docs, 9,225 sentences, 1,798 comparative, 2,468 quintuples | Train/evaluate comparative quintuple extraction | ⏳ request from VLSP organisers (data agreement); fallback below |
| D3 | **Own crawl**: product-linked reviews from thegioididong.com (same source as ViSFD) + cellphones.com.vn | target 40–80 phone models, 20k–60k reviews | Inference corpus for the ranking | ⏳ to do: check robots.txt, ≤1 req/s, store text/rating/date/product only (no usernames) |
| D4 | **Benchmarks**: GSMArena, DXOMARK, AnTuTu/Geekbench | scores for the same 40–80 phones | Ground truth for ranking | ⏳ manual spreadsheet |
| D5 | **Gold comparative test set** from D3 | 300–500 sentences, 2 annotators, Cohen's κ | Measures extraction quality in *our* domain | ⏳ |

Fallback if VCOM isn't granted in time: label comparative sentences from D3 using an LLM
(silver labels) plus the rule baseline, and hand-check D5 as the test set.

Product set = phones with ≥ 50 reviews **and** at least one benchmark available. Phones without
benchmarks stay in the graph (they carry comparison edges) but are excluded from the correlation.

---

## 5. Experiments

| # | Experiment | Methods | Metric |
|---|---|---|---|
| E1 | ABSA on ViSFD | TF-IDF+LR (✅) → PhoBERT-base / ViSoBERT fine-tune → (opt.) PhoBERT-large | aspect detection micro-F1; aspect#polarity micro-F1 |
| E2 | Comparative sentence detection | keyword rules (L06-03) → PhoBERT classifier | P/R/F1 on VCOM + D5 |
| E3 | Quintuple extraction (subject, object, aspect, predicate, type) | rules → ViT5-base seq2seq (generate tuple string) → LLM few-shot (Qwen2.5-7B) | exact/partial-match F1 (VLSP metric) |
| E4 | Entity linking of product mentions | alias dict + fuzzy match vs. crawled catalogue | accuracy on 200 hand-checked mentions |
| E5 | **Ranking** per aspect | sentiment-only, win-rate, Bradley–Terry, PageRank, PageRank/deg, PPR, random-walker, **pRank (Zhang 2010)**, **BT+sentiment fusion** | Spearman ρ, Kendall τ, NDCG@10 vs. D4, bootstrap 95% CI, overall + per price tier |
| E6 | Ablations | gold vs. predicted comparisons (on VCOM-style data); damping factor d; teleport strength γ; edge confidence on/off; spam filter on/off | Δρ |

---

## 6. Current results (for the progress report)

**E1 PhoBERT-base-v2 multi-task** (shared encoder + 4-way head per aspect, pyvi word segmentation, 6 epochs,
best epoch picked on dev). Code: [src/absa_transformer.py](src/absa_transformer.py). ViSFD test set:

| Model | Aspect detection micro-F1 | Aspect#polarity micro-F1 |
|---|---|---|
| TF-IDF + LR | 0.829 | 0.710 |
| **PhoBERT-base-v2** | **0.890** | **0.815** |

Per aspect (aspect#polarity F1): BATTERY 0.856, CAMERA 0.870, PERFORMANCE 0.829, SCREEN 0.771.
PhoBERT-large, XLM-R and 3-seed runs are queued on Kaggle ([configs/absa_suite.json](configs/absa_suite.json)).

**E5 synthetic ranker study** ([results/ranking_synthetic.txt](results/ranking_synthetic.txt), 30 seeds). Spearman ρ
vs. hidden quality as direct sentiment becomes price-relative (λ):

| Ranker | λ=0 | λ=0.5 | λ=0.8 | λ=1 |
|---|---|---|---|---|
| sentiment only | 0.79 | 0.56 | 0.24 | −0.01 |
| pRank (Zhang 2010) | 0.79 | 0.56 | 0.25 | 0.01 |
| pRank, edges only | 0.31 | 0.31 | 0.31 | 0.31 |
| PageRank | 0.25 | 0.25 | 0.25 | 0.25 |
| Bradley–Terry | 0.65 | 0.65 | 0.65 | 0.65 |
| **BT + sentiment (z-fusion)** | 0.76 | **0.68** | **0.54** | **0.43** |

**E3 rule baseline** ([src/compare_rules.py](src/compare_rules.py)): 3,051 / 11,122 ViSFD comments contain a
comparative cue, and the rules extract a tuple from 510. Low recall motivates the LLM extractor (queued on Kaggle).

### Earlier baseline notes

**E1 baseline:** TF-IDF (word 1–2g + char 2–5g) + logistic regression, one model per aspect, ViSFD test set (2,224
comments). Code: [src/absa_baseline.py](src/absa_baseline.py). Output: [results/](results/).

| Metric | Score |
|---|---|
| Aspect detection micro-F1 | **0.829** (macro 0.792) |
| Aspect#polarity micro-F1 | **0.710** |
| Detection F1 for BATTERY / CAMERA / PERFORMANCE / SCREEN | 0.947 / 0.920 / 0.877 / 0.754 |

The aspects with benchmarks are among the easiest to detect, which is good for the ranking stage.
STORAGE (27 test examples) and OTHERS are weak, but neither is ranked.

**E5 synthetic sanity check:** see D1 above. Output: [results/ranking_synthetic.txt](results/ranking_synthetic.txt).

---

## 7. Resources needed

**Compute**

| Job | Where | Estimate |
|---|---|---|
| TF-IDF baselines, rules, ranking, evaluation | Laptop (Apple M5, 16 GB) | ✅ ~1 min per run |
| PhoBERT-base / ViSoBERT fine-tune (≈135M params, 7.8k examples, len 256) | Laptop (PyTorch MPS) or Kaggle | ~tens of minutes per run on a T4-class GPU; slower on MPS. Measure the first epoch |
| PhoBERT-large, ViT5-base seq2seq | **Kaggle Notebooks** (free GPU quota, ~30 h/week) or Colab | <1–2 h per run |
| LLM silver labels / zero-shot baseline | Qwen2.5-7B-Instruct 4-bit via Ollama/MLX on the laptop (~5 GB RAM), or a paid API | Time it on 100 sentences first, then extrapolate to the ~3–5k comparative candidates |
| Crawling 20k–60k reviews | Laptop, throttled | a few hours, run overnight |
| Storage | — | < 1 GB total |

**Software:** Python 3.11 (uv venv at `.venv`), `datasets`, `scikit-learn`, `transformers`, `torch`,
`pyvi` or VnCoreNLP (Java, needed for PhoBERT word segmentation; ViSoBERT doesn't need it), `networkx` (plots),
`requests`/`playwright` (crawling), `scipy`.

**People/time:** ~6–8 h per annotator for D5, plus ~4 h to build the benchmark sheet.

**Accounts:** Hugging Face (token for faster downloads), Kaggle (GPU), VLSP data request for VCOM.

---

## 8. Timeline (final report/presentation date TBD; assumes ~4 weeks)

| Week | Deliverables |
|---|---|
| W1 (24–30/11) | Request VCOM; crawler + product catalogue; benchmark spreadsheet; PhoBERT/ViSoBERT ABSA (E1) |
| W2 (1–7/12) | Comparative detection + rule extractor (E2, E3-rules); entity linking (E4); annotate D5 |
| W3 (8–14/12) | Learned extractor (ViT5/LLM, E3); run full pipeline on D3 → graphs → rankings (E5) |
| W4 (15–21/12) | Ablations (E6), price-tier analysis, visualisations (L03), final report + slides + demo |

Optional system demo: a small Streamlit page to pick an aspect → ranked phones + example review evidence per edge.

## 9. Risks

| Risk | Mitigation |
|---|---|
| VCOM access delayed | LLM silver labels + rules; D5 gold for evaluation |
| Graph too sparse for some aspects | PPR with sentiment prior; merge model variants (A50/A50s) only when justified; report coverage |
| Benchmark coverage low for budget phones (DXOMARK) | Prefer GSMArena/AnTuTu; evaluate per aspect on whatever overlap exists; report n and CI |
| Site blocks crawling / ToS | Throttle, respect robots.txt, use several sources; ViSFD-era reviews as backup |
| Reviews are price-relative | Within-tier and partial correlations (§3, decision D5) |

---

## 10. Update 23/11 (evening): main contribution = EDR

**Expectation–Disconfirmation Ranking (EDR)** ([src/joint_model.py](src/joint_model.py)) is grounded in Oliver (1980): satisfaction = performance − expectation.
Latent θ_{p,a} explains (1) comparisons (noise-aware Bradley–Terry, entity- or brand-level) and (2) ABSA counts,
P(+) = σ(α_a θ_{p,a} − β_a·x_p + b_a), where x = expectation covariates (price, brand) and β_a is learned. It has a hierarchical aspect prior and is fitted by MAP with bootstrap rank intervals.

Study ([src/study_advanced.py](src/study_advanced.py) → [results/study_advanced.txt](results/study_advanced.txt)):
- EDR > pRank / BT+sentiment / sentiment / price-only in every setting with ε ≤ 0.25.
- **Break-even:** comparative mining beats the price heuristic only while the direction error ε ≲ 0.30 (BT+sent: ≈ 0.05).
- ε cannot be learned from the data (confounded with the quality scale). Measure it on gold annotations.
- Brand expectation helps with plentiful, clean data and overfits when data is sparse or noisy. Brand-level comparisons: no measurable gain (negative result).

Real data ([src/aspect_importance.py](src/aspect_importance.py) → [results/aspect_importance.txt](results/aspect_importance.txt)):
stars ~ aspect polarity, R² = 0.615. Negatives cost 1.5–2.7× more than positives add (loss aversion), DESIGN excepted.
Use these as aspect weights for an overall ranking.

Next smart steps once the LLM labels and the crawl are in:
1. Measure ε of each extractor on gold → place it on the break-even curve.
2. Fit EDR on the crawl, report β_a per aspect (how price-relative reviewers are), held-out comparison accuracy, and correlation with benchmarks.
3. Use calibrated extractor probabilities as edge confidence (`conf`).

## 11. Real-data pipeline status (23/11 evening)

The pipeline runs end to end on CPU: [src/pipeline_real.py](src/pipeline_real.py) (crawl → PhoBERT ABSA → rule comparisons →
[src/entity_link.py](src/entity_link.py) → graph) and [src/real_rank.py](src/real_rank.py) (all rankers, benchmark evaluation, held-out check).

**Finding: buyers rarely compare competitors head-to-head.** In 1,325 thegioididong reviews of 2023–26 phones, 21 comparisons are linked:
9 are against the reviewer's *previous* phone ("thua xa reno 2f ngày xưa"), 8 are against a *brand* ("không rực bằng ss"), and only 1 is phone-vs-phone in the catalogue.
→ the buyer's reference point is their old phone or brand (reference dependence again, consistent with EDR).
→ brand-level (set) comparisons matter in real data (≈40% of edges), unlike in the simulation.
→ head-to-head comparisons must come from **expert text**: VnExpress phone articles are being crawled
([src/crawl_vnexpress.py](src/crawl_vnexpress.py); robots allow) and VCOM (VLSP, request needed).
Tinhte blocks Claude's crawler (robots: ClaudeBot Disallow), so we do not crawl it.

**Data sparsity:** thegioididong only lists current phones (discontinued pages removed). Only 22–23 models have ≥ 20 reviews.
Options: add another store (CellphoneS / FPT Shop; needs the team's OK), and pool all storage/colour variants (done via modelId).

**Team tasks:** fill [data/benchmarks_template.csv](data/benchmarks_template.csv) (GSMArena battery hours, DXOMARK camera/display,
AnTuTu) for the top-reviewed phones; request VCOM from VLSP organisers.

## 12. EDR — final form (23/11 night) — [src/edr.py](src/edr.py), study: [src/study_edr_v2.py](src/study_edr_v2.py) → [results/study_edr_v2.txt](results/study_edr_v2.txt)

Three ideas, every parameter interpretable:
1. **Hedonic expectation** ê_{p,a} = x_p·c_a, where x = (z log-price, brand). It is also an empirical-Bayes prior θ_{p,a} ~ N(μ_p + ê_{p,a}, τ²):
   phones with little evidence shrink toward what their price/brand predicts.
2. **Disconfirmation** (Oliver 1980): sentiment P(+) = σ(α_a(θ_{p,a} − γ_a ê_{p,a}) + b_a). γ ∈ [0,1] is the *price-relativity index*.
   Comparisons: noise-aware Bradley–Terry with a measured per-source flip rate ε_s; brand-only mentions are set comparisons.
3. **Comparisons referee the sentiment model.** γ is only weakly identified when learned jointly (γ̂ ≈ 0.64–0.68 even when the truth is 0!).
   Head-to-head comparisons are not price-relative, so γ is chosen by **K-fold held-out comparison log-likelihood**
   (`select_gamma`). Chosen γ tracks the truth: 0.17 / 0.47 / 0.61 for true 0 / 0.5 / 0.8.
4. **Laplace posterior** over θ → calibrated P(A ≻ B) (ECE 0.06–0.10) and credible rank intervals.

Spearman ρ vs true quality, 20 seeds per world:

| World | price | sentiment | pRank | BT+sent | EDR v1 | **EDR final** | γ=0 ablation | −hedonic prior |
|---|---|---|---|---|---|---|---|---|
| strongly price-relative (γ=.8) | .634 | .692 | .700 | .679 | .805 | **.832** | .700 | .820 |
| moderately price-relative (γ=.5) | .634 | .790 | .798 | .725 | .813 | **.837** | .808 | .822 |
| absolute (γ=0): no-harm check | .634 | .838 | .841 | .746 | .812 | **.855** | .856 | .829 |
| sparse (100 comps, γ=.8) | .634 | .691 | .694 | .624 | .757 | **.797** | .661 | .787 |

Dropped after testing (Occam): loss aversion κ in the ranking model (no ranking gain; κ only partly recoverable from counts,
so it stays as a *descriptive* finding from the star regression); the value-for-money output (the raw sentiment ratio is a
better value-for-money signal → insight: **buyer sentiment answers "best value?", benchmarks answer "best?"**).

Real-data note: rules on expert (VnExpress) text are too crude (e.g. "giá cao hơn" = more expensive ≠ better; weight compared
under PERFORMANCE). Expert comparisons need the LLM extractor, which is exactly what the break-even analysis says matters.
