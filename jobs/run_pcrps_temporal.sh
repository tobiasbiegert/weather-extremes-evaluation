#!/usr/bin/env bash
set -eu

# Change to the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load machine-specific directory paths from local_paths.env.
if [ -f "${REPO_ROOT}/local_paths.env" ]; then
  set -a
  source "${REPO_ROOT}/local_paths.env"
  set +a
else
  echo "Missing local_paths.env in ${REPO_ROOT}"
  exit 1
fi

# Save log
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_pcrps_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"

# -------------------------
# Fixed settings
# -------------------------
TIME_START="2022-01-01T12:00:00"
TIME_STOP="2022-12-21T12:00:00"

LEAD_TIMES_HOURS="24,72,120,168,240"

TW_PCRPS="True"

TEMPORAL_MEAN="False"

DIRECT_WORKERS=40
DIRECT_MODE="multi_processing"

TOTAL_JOBS=16
JOB=0

# /dev/null for quiet console
BEAM_OUT="/dev/null"

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

PCRPS_RESULTS_DIR_240X121="${RESULTS_DIR}/pcrps/240x121_t2017_y2022"
mkdir -p "${PCRPS_RESULTS_DIR_240X121}"

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

PCRPS_RESULTS_DIR_0P25="${RESULTS_DIR}/pcrps/0p25_t2017_y2022"
mkdir -p "${PCRPS_RESULTS_DIR_0P25}"

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
  MAXREC_PATH=${7:-}
  MINREC_PATH=${8:-}

  # Skip if output Zarr already exists
  if [ -d "${OUT}" ]; then
    echo ""
    echo "============================================================"
    echo "SKIP  Job ${IDX}/${TOTAL_JOBS}: ${NAME} (output exists)"
    echo "Output:   ${OUT}"
    echo "============================================================"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "START Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Started:  $(date '+%F %T')"
  echo "Output:   ${OUT}"
  echo "============================================================"

  START=$(date +%s)

  CMD=(
    python -m pcrps.compute_pcrps
    --prediction_path="${PRED}"
    --target_path="${TARGET}"
    --output_path="${OUT}"
    --variables="${VARS}"
    --time_start="${TIME_START}"
    --time_stop="${TIME_STOP}"
    --chunk_size_lon="${CHUNK_LON}"
    --chunk_size_lat="${CHUNK_LAT}"
    --tw_pcrps="${TW_PCRPS}"
    --temporal_mean="${TEMPORAL_MEAN}"
    --lead_times_hours="${LEAD_TIMES_HOURS}"
  )

  if [ -n "${MAXREC_PATH}" ]; then
    CMD+=(--monthly_max_records_path="${MAXREC_PATH}")
  fi

  if [ -n "${MINREC_PATH}" ]; then
    CMD+=(--monthly_min_records_path="${MINREC_PATH}")
  fi

  CMD+=(
    --runner=DirectRunner
    --
    --direct_num_workers="${DIRECT_WORKERS}"
    --direct_running_mode="${DIRECT_MODE}"
  )

  # Run the command and hide its output.
  "${CMD[@]}" > "${BEAM_OUT}" 2>&1

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

# -------------------------
# Jobs: 240x121
# -------------------------
# HRES
JOB=$((JOB+1))
run_job "${JOB}" "HRES T2M vs IFS analysis, 240x121" \
  "${HRES_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/hres_240x121_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_240X121}" \
  "${MINREC_240X121}"

JOB=$((JOB+1))
run_job "${JOB}" "HRES WS10 vs IFS analysis, 240x121" \
  "${HRES_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/hres_240x121_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_240X121}"

# Pangu operational
JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational T2M vs IFS analysis, 240x121" \
  "${PANGU_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/pangu_operational_240x121_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_240X121}" \
  "${MINREC_240X121}"

JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational WS10 vs IFS analysis, 240x121" \
  "${PANGU_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/pangu_operational_240x121_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_240X121}"

# GraphCast operational
JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational T2M vs IFS analysis, 240x121" \
  "${GRAPHCAST_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/graphcast_operational_240x121_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_240X121}" \
  "${MINREC_240X121}"

JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational WS10 vs IFS analysis, 240x121" \
  "${GRAPHCAST_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/graphcast_operational_240x121_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_240X121}"

# Aurora
JOB=$((JOB+1))
run_job "${JOB}" "Aurora T2M vs IFS analysis, 240x121" \
  "${AURORA_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/aurora_240x121_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_240X121}" \
  "${MINREC_240X121}"

JOB=$((JOB+1))
run_job "${JOB}" "Aurora WS10 vs IFS analysis, 240x121" \
  "${AURORA_PRED_240X121}" \
  "${IFS_TARGET_240X121}" \
  "${PCRPS_RESULTS_DIR_240X121}/aurora_240x121_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_240X121}"

# -------------------------
# 0p25 settings
# -------------------------
CHUNK_LON=30
CHUNK_LAT=30

# -------------------------
# Jobs: 0p25
# -------------------------
# HRES
JOB=$((JOB+1))
run_job "${JOB}" "HRES T2M vs IFS analysis, 0p25" \
  "${HRES_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/hres_0p25_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_0P25}" \
  "${MINREC_0P25}"

JOB=$((JOB+1))
run_job "${JOB}" "HRES WS10 vs IFS analysis, 0p25" \
  "${HRES_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/hres_0p25_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_0P25}"

# Pangu operational
JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational T2M vs IFS analysis, 0p25" \
  "${PANGU_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/pangu_operational_0p25_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_0P25}" \
  "${MINREC_0P25}"

JOB=$((JOB+1))
run_job "${JOB}" "Pangu operational WS10 vs IFS analysis, 0p25" \
  "${PANGU_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/pangu_operational_0p25_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_0P25}"

# GraphCast operational
JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational T2M vs IFS analysis, 0p25" \
  "${GRAPHCAST_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/graphcast_operational_0p25_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_0P25}" \
  "${MINREC_0P25}"

JOB=$((JOB+1))
run_job "${JOB}" "GraphCast operational WS10 vs IFS analysis, 0p25" \
  "${GRAPHCAST_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/graphcast_operational_0p25_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_0P25}"

# Aurora
JOB=$((JOB+1))
run_job "${JOB}" "Aurora T2M vs IFS analysis, 0p25" \
  "${AURORA_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/aurora_0p25_vs_ifs_analysis_t2m_temporal.zarr" \
  "2m_temperature" \
  "${MAXREC_0P25}" \
  "${MINREC_0P25}"

JOB=$((JOB+1))
run_job "${JOB}" "Aurora WS10 vs IFS analysis, 0p25" \
  "${AURORA_PRED_0P25}" \
  "${IFS_TARGET_0P25}" \
  "${PCRPS_RESULTS_DIR_0P25}/aurora_0p25_vs_ifs_analysis_ws10_temporal.zarr" \
  "10m_wind_speed" \
  "${MAXREC_0P25}"


echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"
