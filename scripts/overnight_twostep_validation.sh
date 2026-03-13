#!/usr/bin/env bash
# Overnight diagnostic run: 10 CH cases × 3 runs with two-step judge (v3 prompt).
# Usage: bash scripts/overnight_twostep_validation.sh
#
# Runs sequentially (one case at a time) with concurrency=1 per case.
# Estimated: ~80 min total.

set -uo pipefail

# Use project venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
ATHENA="${PROJECT_DIR}/.venv/bin/athena"

# Ensure oMLX model is set (short name, not HF name)
export OMLX_MODEL="${OMLX_MODEL:-qwen3.5-35b-a3b-text-hi}"

DATE=$(date +%Y%m%d)
OUTPUT_BASE="${1:-output/overnight-twostep-${DATE}}"
SIM_CONFIG="simulations/smoke-ch-twostep-3.yaml"
CASES_DIR="cases/validation"
LOG_FILE="${OUTPUT_BASE}/run.log"

# All 10 CH validation cases
CASES=(
    ch-247
    ch-1124
    ch-1253
    ch-2461
    ch-3295
    ch-741
    ch-1272
    ch-2358
    ch-2434
    ch-3408
)

mkdir -p "${OUTPUT_BASE}"

echo "================================================================" | tee "${LOG_FILE}"
echo "ATHENA Overnight Two-Step Validation — ${DATE}" | tee -a "${LOG_FILE}"
echo "Cases: ${#CASES[@]}, Runs per case: 3, Sim config: ${SIM_CONFIG}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
echo "================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

TOTAL_START=$(date +%s)
SUCCEEDED=0
FAILED=0
RESULTS_SUMMARY=""

for i in "${!CASES[@]}"; do
    CASE_ID="${CASES[$i]}"
    CASE_FILE="${CASES_DIR}/${CASE_ID}.yaml"
    CASE_OUTPUT="${OUTPUT_BASE}/${CASE_ID}"
    IDX=$((i + 1))

    echo "[${IDX}/${#CASES[@]}] Running ${CASE_ID}..." | tee -a "${LOG_FILE}"
    CASE_START=$(date +%s)

    if "${ATHENA}" run \
        --case "${CASE_FILE}" \
        --simulation "${SIM_CONFIG}" \
        --output "${CASE_OUTPUT}" \
        --concurrency 1 \
        2>&1 | tee -a "${LOG_FILE}"; then

        CASE_END=$(date +%s)
        CASE_ELAPSED=$((CASE_END - CASE_START))
        echo "[${IDX}/${#CASES[@]}] ${CASE_ID}: OK (${CASE_ELAPSED}s)" | tee -a "${LOG_FILE}"
        SUCCEEDED=$((SUCCEEDED + 1))
        RESULTS_SUMMARY="${RESULTS_SUMMARY}${CASE_ID}: OK (${CASE_ELAPSED}s)\n"
    else
        CASE_END=$(date +%s)
        CASE_ELAPSED=$((CASE_END - CASE_START))
        echo "[${IDX}/${#CASES[@]}] ${CASE_ID}: FAILED (${CASE_ELAPSED}s)" | tee -a "${LOG_FILE}"
        FAILED=$((FAILED + 1))
        RESULTS_SUMMARY="${RESULTS_SUMMARY}${CASE_ID}: FAILED (${CASE_ELAPSED}s)\n"
    fi

    echo "" | tee -a "${LOG_FILE}"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))

echo "================================================================" | tee -a "${LOG_FILE}"
echo "SUMMARY" | tee -a "${LOG_FILE}"
echo "================================================================" | tee -a "${LOG_FILE}"
echo -e "${RESULTS_SUMMARY}" | tee -a "${LOG_FILE}"
echo "Succeeded: ${SUCCEEDED}/${#CASES[@]}" | tee -a "${LOG_FILE}"
echo "Failed: ${FAILED}/${#CASES[@]}" | tee -a "${LOG_FILE}"
echo "Total time: ${TOTAL_ELAPSED}s (${TOTAL_MIN} min)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run validation scorer
echo "Running athena validate..." | tee -a "${LOG_FILE}"
"${ATHENA}" validate \
    --results-dir "${OUTPUT_BASE}" \
    --ground-truth ground_truth \
    --output "${OUTPUT_BASE}/validation_report.md" \
    2>&1 | tee -a "${LOG_FILE}"

# Run analysis script
echo "" | tee -a "${LOG_FILE}"
echo "Running analysis..." | tee -a "${LOG_FILE}"
"${VENV_PYTHON}" scripts/analyze_overnight.py \
    --results-dir "${OUTPUT_BASE}" \
    --ground-truth ground_truth \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Done. All outputs in ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
