#!/usr/bin/env bash
# Phase 1, P1-P4: truth for each cell, then replicates, then summaries. Resumable (completed seeds are skipped).
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
for cell in S0 S3_null S3_m070 S3_p020 S1_mod S1_strong S2_mod S2_severe; do
  [ -f results/phase1/truth_$cell.json ] || $PY scenario_study.py truth --cell $cell --workers 10 --chunks 40
done
$PY scenario_study.py run --cell S0 --R 1000 --workers 10
for cell in S3_null S3_m070 S3_p020 S1_mod S1_strong S2_mod S2_severe; do
  $PY scenario_study.py run --cell $cell --R 500 --workers 10
done
for cell in S0 S3_null S3_m070 S3_p020 S1_mod S1_strong S2_mod S2_severe; do
  $PY scenario_study.py summarise --cell $cell
done
echo PHASE1_P1_P4_DONE
