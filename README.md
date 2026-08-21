# ACES — Adaptive Conformal Early-warning System

Reproduction code for:

> Lee, D. *Refresh, forget, or adapt? Online conformal calibration of chlorophyll-a forecasts on a sealed two-basin evaluation.* (manuscript)

The study compares five prediction-interval strategies — raw Gaussian, static split conformal (SCP), expanding conformal (ECP), rolling conformal (RCP), and bounded adaptive conformal inference (ACI) — applied to identical heteroscedastic deep-ensemble chlorophyll-a forecasts at ten automatic monitoring stations in the Geum and Nakdong River basins. All preprocessing and calibration hyperparameters are fixed on data through 2024; the 2025 evaluation year is sealed and read once, by the final evaluation script, after everything is frozen.

## Layout

| Path | Role |
|---|---|
| `src/` | Data adapter (mask-aware 27-channel inputs), backbones (Mamba, GRU, iTransformer) with heteroscedastic heads, training, evaluation utilities |
| `scripts/run_clean_v2_retrain.py` | Trains the 3 backbones × 5 loss arms × 5 replicates × 5 members grid and exports 2024 validation / 2025 test ensemble prediction pairs |
| `scripts/eval_calibration_clean_v2.py` | Validation-only selection of the ACI step size and rolling window, sealed-year five-way evaluation, gate report, and retrospective alert analysis |
| `scripts/make_clean_v2_tables.py`, `scripts/make_clean_v2_figures.py` | Manuscript tables and figures from the evaluation outputs |
| `scripts/run_redesign_retrain.py`, `run_experiment.py`, `run_ablation_loss.py` | Shared training infrastructure (completion sidecars, safe resume, loss-arm configuration) |
| `gis/create_study_area_two_basins.py` | Study-area map; set `ACES_GIS_DIR` to a directory of Korean river-network and provincial-boundary shapefiles |
| `protocols/uq_iso_hab_clean_v2.yml` | Frozen data protocol: station cohort rule, chronological splits (train 2021–2023 / selection 2024 / sealed test 2025), causal input policy, prohibited operations |

## Data

Daily confirmed observations of the national automatic water-quality monitoring network are publicly available from the Water Environment Information System (<https://water.nier.go.kr>). Model arrays are built from those records under `protocols/uq_iso_hab_clean_v2.yml`; the analysis-ready arrays and all prediction outputs accompany the journal submission archive. Point the code at a local bundle with `WEIS_DATA_CORE_ROOT` (adapter) and `WEIS_SAMPLE_INDEX` (evaluation).

## Environment

Python 3.12. Install pinned dependencies with `pip install -r requirements.txt` (choose the PyTorch CUDA build appropriate for your machine). The map script additionally requires `geopandas`.

## Versions

- `v1-ems` — code as released with the earlier single-basin submission (2026-07).
- `v2-wrr` — two-basin, sealed-2025 analysis matching the current manuscript.

## License

MIT — see `LICENSE`.
