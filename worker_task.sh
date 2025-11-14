#!/bin/bash
#
# worker_task.sh
# Helper script executed by Slurm for each worker task via srun.
#
# Arguments:
# $1: START_TIME
# $2: END_TIME
# $3: BP_TABLE
# $4: LUSTRE_WORK_DIR
# $5: PYTHON_SCRIPT
# $6: NVME_RUN_BASE_DIR (e.g., /fast/gh/YYYY-MM-DD/HHh/RUN_NAME/)
# $7: RUN_NAME (e.g., archive_run_TIMESTAMP)
# $8: PEEL_SKY_ARG (Optional)
# $9: PEEL_RFI_ARG (Optional)
# $10: INPUT_DIR (Optional, path or "")
# $11: INTERVALS_ARG (Optional)
# $12: PYTHON_SCRIPT_DIR (NEW)
# $13: PEEL_SKY_MODEL_PATH (NEW)
# $14: PEEL_RFI_MODEL_PATH (NEW)
#

# --- Basic Setup ---
TASK_ID=${SLURM_ARRAY_TASK_ID}
HOSTNAME=$(hostname)

# --- Log Node Allocation ---
echo "Worker task ${TASK_ID} executing on node ${HOSTNAME}"

START_TIME="$1"
END_TIME="$2"
BP_TABLE="$3"
LUSTRE_WORK_DIR="$4"
PYTHON_SCRIPT="$5"
NVME_RUN_BASE_DIR="$6"
RUN_NAME="$7"
PEEL_SKY_ARG="$8"
PEEL_RFI_ARG="$9"
INPUT_DIR="${10}"
INTERVALS_ARG="${11}"
PYTHON_SCRIPT_DIR="${12}" # New
PEEL_SKY_MODEL="${13}"   # New
PEEL_RFI_MODEL="${14}"   # New

LUSTRE_QA_DIR="${LUSTRE_WORK_DIR}/QA"
LUSTRE_RECEIPT_DIR="${LUSTRE_QA_DIR}/receipts"

# Construct input dir argument for python
INPUT_DIR_ARG=""
if [ -n "$INPUT_DIR" ]; then
    INPUT_DIR_ARG="--input-dir ${INPUT_DIR}"
fi

# Basic check for arguments
if [ -z "$RUN_NAME" ] || [ -z "$NVME_RUN_BASE_DIR" ] || [ -z "$PYTHON_SCRIPT_DIR" ]; then
    echo "ERROR (Array Task ${TASK_ID}): Missing arguments (RUN_NAME, NVME_RUN_BASE_DIR, or PYTHON_SCRIPT_DIR) to worker_task.sh" >&2
    exit 1
fi

# --- Create Node-Local NVMe TASK Directory ---
NVME_TASK_DIR="${NVME_RUN_BASE_DIR}/task${TASK_ID}"
echo "Worker array task ${TASK_ID}: NVMe Base Directory: ${NVME_RUN_BASE_DIR}"
echo "Worker array task ${TASK_ID}: Creating Task Directory: ${NVME_TASK_DIR}"
mkdir -p "${NVME_RUN_BASE_DIR}"
mkdir -p "${NVME_TASK_DIR}/QA"
mkdir -p "${NVME_TASK_DIR}/ms_data"

# --- Run Python Script (Orchestrator) ---
echo "Worker array task ${TASK_ID}: Attempting conda run (Mode: subband_work)..."
conda run -n py38_orca_nkosogor python "${PYTHON_SCRIPT}" \
    --mode subband_work \
    --task_id "${TASK_ID}" \
    --start_time "${START_TIME}" \
    --end_time "${END_TIME}" \
    --bandpass_table "${BP_TABLE}" \
    --working_dir "${NVME_TASK_DIR}" \
    --lustre_receipt_dir "${LUSTRE_RECEIPT_DIR}" \
    --script_dir "${PYTHON_SCRIPT_DIR}" \
    --peel_sky_model "${PEEL_SKY_MODEL}" \
    --peel_rfi_model "${PEEL_RFI_MODEL}" \
    ${PEEL_SKY_ARG} \
    ${PEEL_RFI_ARG} \
    ${INPUT_DIR_ARG} \
    ${INTERVALS_ARG} 

PYTHON_EXIT_CODE=$?

# --- Cleanup ---
# The Python script now orchestrates its own sub-jobs.
# This main script simply needs to wait for the orchestrator to finish.
# The local NVMe directory MUST be left intact for the Stage 3
# and Final Imager (scp) jobs.
# We will only copy the *main orchestrator log* back to Lustre.

if [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "Worker array task ${TASK_ID}: Python orchestrator failed with exit code ${PYTHON_EXIT_CODE}" >&2
    PYTHON_LOG=$(ls -t "${NVME_TASK_DIR}/QA/archive_processing_S1_Orchestrator_"* 2>/dev/null | head -n 1)
    if [ -f "${PYTHON_LOG}" ]; then
        echo "Worker array task ${TASK_ID}: Copying failed log ${PYTHON_LOG} to ${LUSTRE_QA_DIR}"
        cp "${PYTHON_LOG}" "${LUSTRE_QA_DIR}/"
    fi
    echo "Worker array task ${TASK_ID}: Python script failed. Leaving NVMe task directory ${NVME_TASK_DIR} for debugging."
    exit ${PYTHON_EXIT_CODE}
fi

echo "Worker array task ${TASK_ID}: Python orchestrator script finished successfully."
echo "It has submitted its own Stage 2 (Peel) and Stage 3 (Gather) jobs."
PYTHON_LOG=$(ls -t "${NVME_TASK_DIR}/QA/archive_processing_S1_Orchestrator_"* 2>/dev/null | head -n 1)
if [ -f "${PYTHON_LOG}" ]; then
    echo "Worker array task ${TASK_ID}: Copying log ${PYTHON_LOG} to ${LUSTRE_QA_DIR}"
    cp "${PYTHON_LOG}" "${LUSTRE_QA_DIR}/"
fi

echo "Worker array task ${TASK_ID}: Helper script finished. NVMe directory ${NVME_TASK_DIR} is being left for subsequent jobs."
exit 0
