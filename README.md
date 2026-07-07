# ACES — Adaptive Conformal Early-warning System

Code accompanying the paper *"Calibration, not loss design, governs
prediction-interval quality: an adaptive conformal early-warning system for
algal-bloom forecasting."*

The study runs a pre-registered benchmark that crosses five heteroscedastic
loss families with three forecasting backbones (Mamba, GRU, inverted
Transformer), pairs every trained predictor with raw, static split-conformal,
and online adaptive conformal calibration, and evaluates an early-warning
decision layer built on the calibrated intervals. The central finding is that
the calibration layer, not the training loss, governs the quality of the
delivered prediction interval under distribution shift.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed on Python 3.11 with PyTorch (MPS/CUDA/CPU). A GPU is recommended for
training but not required for the calibration and evaluation stages.

## Data

The study uses publicly available observations from the Korean national
water-quality monitoring network (Water Environment Information System,
<https://water.nier.go.kr>), operated by the National Institute of
Environmental Research: daily records for five Geum River basin stations
(Gongju, Daecheongho, Gapcheon, Buyeo, Yongdamho), 2021–2024.

Raw records are not redistributed here. `src/data/preprocessor.py` reconstructs
the analysis-ready daily series (target: chlorophyll-a, log1p; engineered
lag/moving-average features) and expects `data/imputed_daily_data.csv` with the
station columns described in that module. Train 2021–2022 / validation 2023 /
test 2024.

## Reproducing the results

Training produces the deep ensembles; the remaining steps are inference and
evaluation only and do not retrain.

```bash
# 1. Train the M=5 deep ensembles for every backbone x loss arm (and replicates)
python scripts/run_replicates.py --reps 1 2 3 4

# 2. Dump 2023-validation predictions for the saved ensembles
python scripts/dump_val_predictions.py --reps 0 1 2 3 4 5

# 3. ACP benchmark: raw vs split-conformal vs online ACI
python scripts/eval_acp_benchmark.py --alpha 0.10 --gamma 0.02 --calib-source val

# 4. Multi-arm loss statistics (Friedman + Holm + TOST) and method comparison
python scripts/analyze_arms_multi.py

# 5. Operational decision-layer evaluation
python scripts/operational_eval.py

# 6. Figures and consolidated results
python scripts/make_paper_figures.py
python scripts/make_decision_figures.py
python scripts/consolidate_results.py
```

`scripts/run_final_analysis.sh` chains the inference-through-consolidation
stages end-to-end once the ensembles from step 1 exist. Outputs are written
under `results/acp_benchmark/` (`RESULTS.md`, figures, and CSVs).

## Repository layout

```
src/
  data/           # preprocessing and the station data pipeline
  models/         # backbones, decoupled mean/variance towers, loss families
  training/       # trainer with the warmup -> penalty curriculum
  evaluation/     # metrics, conformal calibration (OnlineACP), proper scores, SHAP
  visualization/  # shared paper figure style and plotting helpers
scripts/          # training driver, benchmark, statistics, figures, SHAP
requirements.txt
```

## Loss families

`src/models/losses.py` implements the five pre-registered arms: standard
Gaussian NLL, β-NLL, faithful heteroscedastic regression, and ISO-NLL (a
control that penalises the correlation between the predicted mean and
variance). ISO-NLL is a pre-registered negative control, not a proposed method.

## License

Released under the MIT License (see `LICENSE`).

## Citation

A BibTeX entry will be added once the paper is published. Until then, please
cite the manuscript by title and the author, Donghyun Lee (Hankuk University of
Foreign Studies).

## Acknowledgements

This work was partly supported by the Institute of Information & Communications
Technology Planning & Evaluation (IITP) grant funded by the Korea government
(MSIT) (No. RS-2026-25522834, Development of an Ultra-Fast, High-Reliability
Physics-AI Hybrid Disaster Response Technology Based on Mamba-Flow Matching), by
the National Research Foundation of Korea (NRF) grant funded by the Korea
government (MSIT) (No. RS-2026-25493931), and by the Hankuk University of Foreign
Studies Research Fund of 2026.
