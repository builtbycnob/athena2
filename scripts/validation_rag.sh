#!/usr/bin/env bash
# Validation run: 10 CH cases × 3 runs WITH RAG enabled.
# Compares against baseline (validation-sev-full, 80% accuracy).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
ATHENA="${PROJECT_DIR}/.venv/bin/athena"

export OMLX_MODEL="${OMLX_MODEL:-qwen3.5-35b-a3b-text-hi}"
export ATHENA_RAG_ENABLED=1

DATE=$(date +%Y%m%d)
OUTPUT_BASE="${1:-output/validation-rag-${DATE}}"
SIM_CONFIG="simulations/smoke-ch-twostep-3.yaml"
CASES_DIR="cases/validation"
LOG_FILE="${OUTPUT_BASE}/run.log"

CASES=(ch-247 ch-1124 ch-1253 ch-2461 ch-3295 ch-741 ch-1272 ch-2358 ch-2434 ch-3408)

mkdir -p "${OUTPUT_BASE}"

echo "================================================================" | tee "${LOG_FILE}"
echo "ATHENA RAG Validation — ${DATE}" | tee -a "${LOG_FILE}"
echo "Cases: ${#CASES[@]}, Runs per case: 3, RAG: ENABLED" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
echo "================================================================" | tee -a "${LOG_FILE}"

TOTAL_START=$(date +%s)
SUCCEEDED=0
FAILED=0
RESULTS_SUMMARY=""

for i in "${!CASES[@]}"; do
    CASE_ID="${CASES[$i]}"
    CASE_FILE="${CASES_DIR}/${CASE_ID}.yaml"
    CASE_OUTPUT="${OUTPUT_BASE}/${CASE_ID}"
    IDX=$((i + 1))

    echo "[${IDX}/${#CASES[@]}] Running ${CASE_ID} (RAG)..." | tee -a "${LOG_FILE}"
    CASE_START=$(date +%s)

    if "${ATHENA}" run \
        --case "${CASE_FILE}" \
        --simulation "${SIM_CONFIG}" \
        --output "${CASE_OUTPUT}" \
        --rag \
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

# Validation scoring
echo "" | tee -a "${LOG_FILE}"
echo "Running athena validate..." | tee -a "${LOG_FILE}"
"${ATHENA}" validate \
    --results-dir "${OUTPUT_BASE}" \
    --ground-truth ground_truth \
    --output "${OUTPUT_BASE}/validation_report.md" \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Running analysis..." | tee -a "${LOG_FILE}"
"${VENV_PYTHON}" scripts/analyze_overnight.py \
    --results-dir "${OUTPUT_BASE}" \
    --ground-truth ground_truth \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Done. All outputs in ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
