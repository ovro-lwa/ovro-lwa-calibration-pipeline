#!/bin/bash
#
# submit_archive_processing.sh
#
# Master script to orchestrate the 4-STAGE (Scatter-Gather-Aggregate)
# processing of the night-time archive data using Slurm.
#
# 1. Submits a 16-task array (Job 1) to process each sub-band
#    on its own node, including local peeling and concatenation.
# 2. Submits a single high-memory job (Job 2) that depends on Job 1.
#    Job 2 aggregates all 16 concatenated MS files and
#    runs the final deep UV-Join imaging.
#
# CHANGELOG: Replaced buggy getopt parser with a robust BASH while loop
#            to fix the datetime string error.
#

# --- 1. Argument Parsing (NEW Robust Parser) ---
START_TIME=""
END_TIME=""
BP_TABLE=""
INPUT_DIR=""
IMAGER_NODE=""
DO_PEELING=0
DO_PEELING_RFI=0
PEELING_ARG=""
PEELING_RFI_ARG=""
INPUT_DIR_ARG=""
INTERVALS_ARG=""
# --- Full paths to model files ---
PEEL_SKY_MODEL="/lustre/gh/calibration/pipeline/workingv14/sources.json"
PEEL_RFI_MODEL="/lustre/gh/calibration/pipeline/workingv14/rfi_43.2_ver20251101.json"

# This loop replaces getopt to avoid quote-mangling errors
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --start-time)
        START_TIME="$2"
        shift; shift
        ;;
        --end-time)
        END_TIME="$2"
        shift; shift
        ;;
        --bandpass-table)
        BP_TABLE="$2"
        shift; shift
        ;;
        --input-dir)
        INPUT_DIR="$2"
        INPUT_DIR_ARG="--input-dir $2"
        shift; shift
        ;;
        --imager-node)
        IMAGER_NODE="$2"
        shift; shift
        ;;
        --peel-sky)
        DO_PEELING=1
        PEELING_ARG="--peel-sky"
        shift
        ;;
        --peel-rfi)
        DO_PEELING_RFI=1
        PEELING_RFI_ARG="--peel-rfi"
        shift
        ;;
        --intervals-out)
        # Handle optional argument for intervals-out
        if [[ -n "$2" ]] && ! [[ "$2" =~ ^-- ]]; then
            INTERVALS_ARG="--intervals-out $2"
            shift; shift
        else
            INTERVALS_ARG="--intervals-out"
            shift
        fi
        ;;
        *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
done
# --- End of new parser ---

# Check for required arguments
if [ -z "$START_TIME" ] || [ -z "$END_TIME" ] || [ -z "$BP_TABLE" ]; then
    echo "Usage: $0 --start-time YYYY-MM-DD:HH:MM:SS --end-time YYYY-MM-DD:HH:MM:SS --bandpass-table /path/to/table.B [options]"
    echo "Optional: --imager-node <hostname> (e.g., lwacalim10)"
    exit 1
fi

if [ $DO_PEELING -eq 1 ] && [ ! -f "$PEEL_SKY_MODEL" ]; then
    echo "ERROR: --peel-sky enabled, but model file not found: $PEEL_SKY_MODEL" >&2; exit 1; fi
if [ $DO_PEELING_RFI -eq 1 ] && [ ! -f "$PEEL_RFI_MODEL" ]; then
    echo "ERROR: --peel-rfi enabled, but model file not found: $PEEL_RFI_MODEL" >&2; exit 1; fi

# --- 2. Configuration & Setup ---

PYTHON_SCRIPT_DIR="$(dirname "$0")"
PYTHON_SCRIPT="${PYTHON_SCRIPT_DIR}/process_archive.py"
HELPER_SCRIPT="${PYTHON_SCRIPT_DIR}/worker_task.sh"
LUSTRE_OUTPUT_BASE_DIR="/lustre/gh/processing"
NVME_BASE_DIR="/fast/gh" 

DATA_DATE=$(echo ${START_TIME} | cut -d':' -f1)
DATA_HOUR=$(echo ${START_TIME} | cut -d':' -f2)
UTC_HOUR_STR="${DATA_HOUR}h"
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="archive_run_${RUN_TIMESTAMP}"

LUSTRE_WORK_DIR="${LUSTRE_OUTPUT_BASE_DIR}/${DATA_DATE}/${UTC_HOUR_STR}/${RUN_NAME}"
LUSTRE_QA_DIR="${LUSTRE_WORK_DIR}/QA"
LUSTRE_RECEIPT_DIR="${LUSTRE_QA_DIR}/receipts" # New dir for aggregation
NVME_RUN_BASE_DIR="${NVME_BASE_DIR}/${DATA_DATE}/${UTC_HOUR_STR}/${RUN_NAME}" # For worker tasks
NVME_IMAGER_DIR="${NVME_BASE_DIR}/${DATA_DATE}/${UTC_HOUR_STR}/${RUN_NAME}_Imager" # For final imager

echo "--- Starting 4-Stage Archive Processing Run: ${RUN_NAME} ---"
echo "Start Time: ${START_TIME}"
echo "End Time:   ${END_TIME}"
echo "BP Table:   ${BP_TABLE}"
if [ -n "$INPUT_DIR" ]; then echo "Input Dir:  ${INPUT_DIR}"; else echo "Input Dir:  (Standard Archive)"; fi
if [ $DO_PEELING -eq 1 ]; then echo "Astro Peeling:  Enabled"; fi
if [ $DO_PEELING_RFI -eq 1 ]; then echo "RFI Peeling:    Enabled"; fi
if [ -n "$INTERVALS_ARG" ]; then echo "Imaging:    ${INTERVALS_ARG}"; fi
if [ -n "$IMAGER_NODE" ]; then 
    echo "Imager Node: **Forced to ${IMAGER_NODE}**"
else
    echo "Imager Node: (Letting Slurm choose)"
fi

echo "Creating final output directory: ${LUSTRE_WORK_DIR}"
mkdir -p "${LUSTRE_QA_DIR}"
mkdir -p "${LUSTRE_RECEIPT_DIR}" # Create receipt directory
echo "Worker NVMe base directory: ${NVME_RUN_BASE_DIR}"
echo "Final Imager NVMe directory: ${NVME_IMAGER_DIR}"

if [ ! -x "${HELPER_SCRIPT}" ]; then
    echo "ERROR: Helper script '${HELPER_SCRIPT}' not found or not executable."
    exit 1
fi
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script '${PYTHON_SCRIPT}' not found."
    exit 1
fi

# --- 3. Submit Workflow ---

# --- Job 1: Worker Array (Peeling & Local Concat) ---
echo "Mode: Multi-Node Scatter-Gather. Submitting worker array..."
echo "Requesting 32 CPUs and 384G RAM for each of 16 tasks."

JOB_NAME="archive_worker"
LOG_OUT="${LUSTRE_QA_DIR}/slurm-${JOB_NAME}-%A_%a.out"
LOG_ERR="${LUSTRE_QA_DIR}/slurm-${JOB_NAME}-%A_%a.err"

# We pass the *script dir* and model paths to the helper script
# The helper script passes them to process_archive.py
ARRAY_JOB_ID_RAW=$(sbatch --job-name=${JOB_NAME} --output=${LOG_OUT} --error=${LOG_ERR} --partition=general \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=32G \
    --time=24:00:00 --array=0-15 \
    --wrap="srun ${HELPER_SCRIPT} \"${START_TIME}\" \"${END_TIME}\" \"${BP_TABLE}\" \
    \"${LUSTRE_WORK_DIR}\" \"${PYTHON_SCRIPT}\" \"${NVME_RUN_BASE_DIR}\" \"${RUN_NAME}\" \
    \"${PEELING_ARG}\" \"${PEELING_RFI_ARG}\" \"${INPUT_DIR}\" \"${INTERVALS_ARG}\" \
    \"${PYTHON_SCRIPT_DIR}\" \"${PEEL_SKY_MODEL}\" \"${PEEL_RFI_MODEL}\"")

ARRAY_JOB_ID=$(echo ${ARRAY_JOB_ID_RAW} | awk '{print $NF}')
if [ -z "${ARRAY_JOB_ID}" ]; then echo "ERROR: Failed to submit worker job array."; exit 1; fi
echo "Submitted worker job array (Job ID: ${ARRAY_JOB_ID})"

# --- Job 2: Final Imager (Aggregation & UV-Join) ---
echo "Submitting final aggregation and imaging job..."
DEP_STRING="afterok:${ARRAY_JOB_ID}"
JOB_NAME="archive_imager"
IMG_LOG_OUT="${LUSTRE_QA_DIR}/slurm-${JOB_NAME}-%j.out"
IMG_LOG_ERR="${LUSTRE_QA_DIR}/slurm-${JOB_NAME}-%j.err"

# --- Set Nodelist Argument if provided ---
IMAGER_NODE_ARG=""
if [ -n "$IMAGER_NODE" ]; then
    IMAGER_NODE_ARG="--nodelist=${IMAGER_NODE}"
fi

# High-memory, multi-cpu job for WSClean
COADD_JOB_ID_RAW=$(sbatch \
    --job-name=${JOB_NAME} \
    --output=${IMG_LOG_OUT} \
    --error=${IMG_LOG_ERR} \
    --partition=general \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=384G \
    --time=24:00:00 \
    --dependency=${DEP_STRING} \
    ${IMAGER_NODE_ARG} \
    --wrap="echo '--- Final Imager Job Starting ---'; \
            mkdir -p \"${NVME_IMAGER_DIR}/QA\"; \
            mkdir -p \"${NVME_IMAGER_DIR}/ms_data\"; \
            conda run -n py38_orca_nkosogor python \"${PYTHON_SCRIPT}\" \
                --mode final_image \
                --working_dir \"${NVME_IMAGER_DIR}\" \
                --lustre_qa_dir \"${LUSTRE_QA_DIR}\" \
                --lustre_receipt_dir \"${LUSTRE_RECEIPT_DIR}\" \
                ${INTERVALS_ARG}; \
            echo '--- Final Imager Job Finished ---'"
)

COADD_JOB_ID=$(echo ${COADD_JOB_ID_RAW} | awk '{print $NF}')
if [ -z "${COADD_JOB_ID}" ]; then echo "ERROR: Failed to submit final imager job."; exit 1; fi
echo "Submitted ${JOB_NAME} (Job ID: ${COADD_JOB_ID}), dependent on ${ARRAY_JOB_ID}"

echo "--- Workflow Submitted ---"
echo "Monitor progress with: squeue -u \${USER}"
echo "Final results will be in: ${LUSTRE_QA_DIR}/"

exit 0
