#!/usr/bin/env bash
set -eu

# Change to repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# MEMORY_LIMIT_GB=175
# ulimit -v $((MEMORY_LIMIT_GB * 1024 * 1024))

# Load machine-specific directory paths from local_paths.env.
if [ -f "${REPO_ROOT}/local_paths.env" ]; then
    set -a
    source "${REPO_ROOT}/local_paths.env"
    set +a
else
    echo "Missing local_paths.env in ${REPO_ROOT}" >&2
    exit 1
fi

# Save log.
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_taggart_decomposition_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"

# -------------------------
# Fixed settings
# -------------------------
TIME_START="2022-01-01T12:00:00"
TIME_STOP="2022-12-21T12:00:00"
EXPECTED_INITIALIZATIONS=708

LEAD_TIMES_HOURS="24,72,120,168,240"

VARS_MAXREC="2m_temperature,10m_wind_speed"
VARS_MINREC="2m_temperature"

# -------------------------
# 240x121 paths
# -------------------------
MAXREC_240X121="${THRESHOLDS_DIR}/era5_240x121_record_max_month_1979_2017.zarr"
MINREC_240X121="${THRESHOLDS_DIR}/era5_240x121_record_min_month_1979_2017.zarr"

IFS_TARGET_240X121="${DATA_DIR}/hres_t0_240x121_2022_eval_times.zarr"

HRES_PRED_240X121="${DATA_DIR}/hres_240x121_2022.zarr"
PANGU_PRED_240X121="${DATA_DIR}/pangu_operational_240x121_2022.zarr"
GRAPHCAST_PRED_240X121="${DATA_DIR}/graphcast_operational_240x121_2022.zarr"
AURORA_PRED_240X121="${DATA_DIR}/aurora_240x121_2022.zarr"

TAGGART_RESULTS_DIR_240X121="${RESULTS_DIR}/taggart/240x121_t2017_y2022"
mkdir -p "${TAGGART_RESULTS_DIR_240X121}"

# -------------------------
# 0p25 paths
# -------------------------
MAXREC_0P25="${THRESHOLDS_DIR}/era5_1440x721_record_max_month_1979_2017.zarr"
MINREC_0P25="${THRESHOLDS_DIR}/era5_1440x721_record_min_month_1979_2017.zarr"

IFS_TARGET_0P25="${DATA_DIR}/hres_t0_0p25_2022_eval_times.zarr"

HRES_PRED_0P25="${DATA_DIR}/hres_0p25_2022.zarr"
PANGU_PRED_0P25="${DATA_DIR}/pangu_operational_0p25_2022.zarr"
GRAPHCAST_PRED_0P25="${DATA_DIR}/graphcast_operational_0p25_2022.zarr"
AURORA_PRED_0P25="${DATA_DIR}/aurora_0p25_2022.zarr"

TAGGART_RESULTS_DIR_0P25="${RESULTS_DIR}/taggart/0p25_t2017_y2022"
mkdir -p "${TAGGART_RESULTS_DIR_0P25}"

TOTAL_JOBS=16
JOB=0

# Check that all local inputs exist before starting the first job.
for INPUT_PATH in \
    "${IFS_TARGET_240X121}" \
    "${HRES_PRED_240X121}" \
    "${PANGU_PRED_240X121}" \
    "${GRAPHCAST_PRED_240X121}" \
    "${AURORA_PRED_240X121}" \
    "${MAXREC_240X121}" \
    "${MINREC_240X121}" \
    "${IFS_TARGET_0P25}" \
    "${HRES_PRED_0P25}" \
    "${PANGU_PRED_0P25}" \
    "${GRAPHCAST_PRED_0P25}" \
    "${AURORA_PRED_0P25}" \
    "${MAXREC_0P25}" \
    "${MINREC_0P25}"
do
    if [ ! -d "${INPUT_PATH}" ]; then
        echo "Missing input Zarr store: ${INPUT_PATH}" >&2
        exit 1
    fi
done

# -------------------------
# Run one job + print minimal status + timing
# -------------------------
run_job () {
    IDX=$1
    NAME=$2
    PRED=$3
    TARGET=$4
    OUT=$5
    VARS=$6
    THRESH=$7
    TAIL=$8

    if [ -d "${OUT}" ]; then
        echo ""
        echo "============================================================"
        echo "SKIP Job ${IDX}/${TOTAL_JOBS}: ${NAME} (output exists)"
        echo "Output: ${OUT}"
        echo "============================================================"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "START Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
    echo "Started: $(date '+%F %T')"
    echo "Output: ${OUT}"
    echo "============================================================"

    START=$(date +%s)

    CMD=(
        python evaluate_taggart_decomposition.py
        --prediction_path="${PRED}"
        --target_path="${TARGET}"
        --threshold_path="${THRESH}"
        --output_path="${OUT}"
        --variables="${VARS}"
        --tail="${TAIL}"
        --time_start="${TIME_START}"
        --time_stop="${TIME_STOP}"
        --expected_initializations="${EXPECTED_INITIALIZATIONS}"
        --chunk_size_lon="${CHUNK_LON}"
        --chunk_size_lat="${CHUNK_LAT}"
        --lead_times_hours="${LEAD_TIMES_HOURS}"
    )

    if [ -n "${NUM_WORKERS}" ]; then
        CMD+=(--num_workers="${NUM_WORKERS}")
    fi

    "${CMD[@]}"

    END=$(date +%s)
    DUR=$((END - START))

    echo "DONE  Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
    echo "Finished: $(date '+%F %T')"
    echo "Duration: ${DUR} seconds"
    echo "============================================================"
}

# -------------------------
# 240x121 settings
# -------------------------
CHUNK_LON=8
CHUNK_LAT=11
NUM_WORKERS=24

# -------------------------
# Jobs: monthly maximum records, 240x121
# Relevant score: S2
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "HRES vs IFS analysis monthly maximum, 240x121" \
    "${HRES_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/hres_240x121_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_240X121}" \
    "upper"

JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational vs IFS analysis monthly maximum, 240x121" \
    "${PANGU_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/pangu_operational_240x121_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_240X121}" \
    "upper"

JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational vs IFS analysis monthly maximum, 240x121" \
    "${GRAPHCAST_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/graphcast_operational_240x121_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_240X121}" \
    "upper"

JOB=$((JOB+1))
run_job "${JOB}" "Aurora vs IFS analysis monthly maximum, 240x121" \
    "${AURORA_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/aurora_240x121_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_240X121}" \
    "upper"

# -------------------------
# Jobs: monthly minimum records, 240x121
# Relevant score: S1
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "HRES vs IFS analysis monthly minimum, 240x121" \
    "${HRES_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/hres_240x121_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_240X121}" \
    "lower"

JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational vs IFS analysis monthly minimum, 240x121" \
    "${PANGU_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/pangu_operational_240x121_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_240X121}" \
    "lower"

JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational vs IFS analysis monthly minimum, 240x121" \
    "${GRAPHCAST_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/graphcast_operational_240x121_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_240X121}" \
    "lower"

JOB=$((JOB+1))
run_job "${JOB}" "Aurora vs IFS analysis monthly minimum, 240x121" \
    "${AURORA_PRED_240X121}" \
    "${IFS_TARGET_240X121}" \
    "${TAGGART_RESULTS_DIR_240X121}/aurora_240x121_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_240X121}" \
    "lower"

# -------------------------
# 0p25 settings
# -------------------------
CHUNK_LON=60
CHUNK_LAT=60

# -------------------------
# Jobs: monthly maximum records, 0p25
# Relevant score: S2
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "HRES vs IFS analysis monthly maximum, 0p25" \
    "${HRES_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/hres_0p25_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_0P25}" \
    "upper"

JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational vs IFS analysis monthly maximum, 0p25" \
    "${PANGU_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/pangu_operational_0p25_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_0P25}" \
    "upper"

JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational vs IFS analysis monthly maximum, 0p25" \
    "${GRAPHCAST_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/graphcast_operational_0p25_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_0P25}" \
    "upper"

JOB=$((JOB+1))
run_job "${JOB}" "Aurora vs IFS analysis monthly maximum, 0p25" \
    "${AURORA_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/aurora_0p25_vs_ifs_analysis_monthly_max_record.zarr" \
    "${VARS_MAXREC}" \
    "${MAXREC_0P25}" \
    "upper"

# -------------------------
# Jobs: monthly minimum records, 0p25
# Relevant score: S1
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "HRES vs IFS analysis monthly minimum, 0p25" \
    "${HRES_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/hres_0p25_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_0P25}" \
    "lower"

JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational vs IFS analysis monthly minimum, 0p25" \
    "${PANGU_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/pangu_operational_0p25_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_0P25}" \
    "lower"

JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational vs IFS analysis monthly minimum, 0p25" \
    "${GRAPHCAST_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/graphcast_operational_0p25_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_0P25}" \
    "lower"

JOB=$((JOB+1))
run_job "${JOB}" "Aurora vs IFS analysis monthly minimum, 0p25" \
    "${AURORA_PRED_0P25}" \
    "${IFS_TARGET_0P25}" \
    "${TAGGART_RESULTS_DIR_0P25}/aurora_0p25_vs_ifs_analysis_monthly_min_record.zarr" \
    "${VARS_MINREC}" \
    "${MINREC_0P25}" \
    "lower"

echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"