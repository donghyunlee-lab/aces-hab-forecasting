#!/usr/bin/env bash
# Final ACP-benchmark analysis — runs the whole pipeline end-to-end in one shot.
# Detached via setsid so it survives session exit. Every step is idempotent
# (re-running is safe): the val dump skips existing files, the evaluators and
# figure scripts just overwrite their CSV/PNG outputs.
#
# Launch (survives session exit):
#   setsid bash scripts/run_final_analysis.sh > final_analysis.log 2>&1 < /dev/null &
#
# Watch:   tail -f final_analysis.log
# Done?:   test -f FINAL_ANALYSIS_DONE && cat $_
#
# Pipeline:
#   1 dump 2023-val predictions (GPU inference, ~mins; skips existing)
#   2 ACP benchmark, calib=val  (canonical fair baseline)
#   3 ACP benchmark, calib=2024head (sanity / demo reproduction)
#   4 gamma sensitivity sweep
#   5 multi-arm loss stats (Friedman + Holm + TOST) + method comparison
#   6 paper figures
#   7 consolidate -> RESULTS.md
set -u
cd "$(dirname "$0")/.." || exit 1   # project root (parent of scripts/)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
ROOT="$(pwd)"
MARK="$ROOT/FINAL_ANALYSIS_DONE"
rc=0

run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "[$(date)] >>> $label"
  "$@"
  local e=$?
  echo "[$(date)] <<< $label exit=$e"
  if [ $e -ne 0 ]; then rc=$e; fi
  return $e
}

echo "[$(date)] FINAL ANALYSIS START (pid $$)"

run "1/7 dump 2023-val predictions" \
    $PY scripts/dump_val_predictions.py --reps 0 1 2 3 4 5

# Only proceed to evaluation if the val dump succeeded (the rest depend on it).
if [ $rc -eq 0 ]; then
  run "2/7 ACP benchmark (calib=val, canonical)" \
      $PY scripts/eval_acp_benchmark.py --alpha 0.10 --gamma 0.02 --calib-source val
  run "3/7 ACP benchmark (calib=2024head, sanity)" \
      $PY scripts/eval_acp_benchmark.py --alpha 0.10 --gamma 0.02 --calib-source 2024head --calib-frac 0.2
  run "4/7 gamma sensitivity sweep" \
      $PY scripts/eval_acp_benchmark.py --gamma-sweep 0.005 0.01 0.02 0.05 0.1
  run "5/7 multi-arm loss statistics" \
      $PY scripts/analyze_arms_multi.py
  run "6/7 paper figures" \
      $PY scripts/make_paper_figures.py
  run "7/7 consolidate -> RESULTS.md" \
      $PY scripts/consolidate_results.py
else
  echo "[$(date)] SKIP steps 2-7 (val dump failed, rc=$rc)"
fi

if [ $rc -eq 0 ]; then
  echo "[$(date)] FINAL ANALYSIS COMPLETE OK" | tee "$MARK"
  echo "  results: $ROOT/results/acp_benchmark/  (RESULTS.md, figures/, *.csv)" | tee -a "$MARK"
else
  echo "[$(date)] FINAL ANALYSIS FINISHED WITH ERRORS rc=$rc" | tee "$MARK"
fi
exit $rc
