#!/usr/bin/env bash
export OMLX_MODEL="qwen3.5-35b-a3b-text-hi"
ATHENA=".venv/bin/athena"
SIM="simulations/smoke-ch-twostep-3.yaml"
OUT="output/overnight-twostep-20260313-sev"
LOG="${OUT}/run.log"

mkdir -p "${OUT}"
echo "=== Severity calibration test ===" > "${LOG}"

for CASE in ch-247 ch-1124 ch-1253 ch-2434; do
  echo "[Running $CASE]" | tee -a "${LOG}"
  "${ATHENA}" run --case "cases/validation/${CASE}.yaml" --simulation "${SIM}" --output "${OUT}/${CASE}" --concurrency 1 2>&1 | tee -a "${LOG}"
  echo "[${CASE} done]" | tee -a "${LOG}"
  echo "" | tee -a "${LOG}"
done

.venv/bin/python scripts/analyze_overnight.py --results-dir "${OUT}" --ground-truth ground_truth 2>&1 | tee -a "${LOG}"
echo "=== DONE ===" | tee -a "${LOG}"
