#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/home/leegyuwon/Documents/task1/logs}"
MANIPTRANS_DIR="${MANIPTRANS_DIR:-/home/leegyuwon/Documents/ManipTrans}"
PYTHON_BIN="${PYTHON_BIN:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin}"
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib}"
WAIT_FOR_PIDS="${WAIT_FOR_PIDS:-}"

mkdir -p "${LOG_DIR}"

for pid in ${WAIT_FOR_PIDS}; do
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[train5000] waiting for existing process pid=${pid} at $(date '+%F %T')"
    sleep 10
  done
done

stamp="$(date +%Y%m%d_%H%M%S)"
train_log="${LOG_DIR}/cap_grasp_train_triangle_xy_reach_z040_palmclear_thumbw2_5000ep_${stamp}.log"

echo "[train5000] starting at $(date '+%F %T')"
echo "[train5000] log=${train_log}"

cd "${MANIPTRANS_DIR}"
exec env \
  PATH="${CONDA_BIN_DIR}:${PATH}" \
  LD_LIBRARY_PATH="${CONDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
  HYDRA_FULL_ERROR=1 \
  CAP_GRASP_MODE=1 \
  CAP_GRASP_TOOLS_DIR=/home/leegyuwon/Documents/task1/tools \
  CAP_GRASP_URDF=/home/leegyuwon/Documents/task1/assets/cap_only/cap_only.urdf \
  CAP_GRASP_LOG_DIR="${LOG_DIR}" \
  CAP_GRASP_LOG_STAMP="triangle_xy_reach_z040_palmclear_thumbw2_5000ep_${stamp}" \
  CAP_GRASP_WRIST_SAMPLER=mcp_reachable \
  CAP_REQUIRE_CAP_CENTER_IN_ZERO_TRIANGLE=1 \
  CAP_PALM_DOWN_MAX_DEG=20 \
  CAP_WRIST_Z_MIN=0.040 \
  CAP_WRIST_Z_MAX=0.095 \
  CAP_REJECT_PALM_COLLISION=1 \
  CAP_PALM_COLLISION_MARGIN=0.002 \
  CAP_THUMB_MCP_OUTSIDE_MARGIN=none \
  CAP_REACH_CONTACT_MARGIN=0.020 \
  CAP_REACH_MIN_FINGERS=2 \
  CAP_REACH_REQUIRE_THUMB=1 \
  CAP_THUMB_REACH_MARGIN=0.025 \
  CAP_REACH_MCP_STEPS=17 \
  CAP_REACH_SEGMENT_SAMPLES=5 \
  CAP_GRASP_WRIST_MAX_TRIES=3000 \
  CAP_GRASP_LOCK_CAP_BEFORE_GRAVITY=1 \
  CAP_GRASP_CAP_LOCK_MODE=teleport \
  CAP_GRASP_RESAMPLE_WRIST_ON_RESET=1 \
  CAP_GRASP_TIME=1.0 \
  CAP_SETTLE_TIME=0.5 \
  CAP_HOLD_TIME=2.0 \
  CAP_GRAVITY_RAMP_TIME=0.0 \
  CAP_THUMB_CONTACT_W=2.0 \
  "${PYTHON_BIN}" main/rl/train.py \
  task=ObjDexUnscrew \
  side=RH \
  dexhand=dg5fs \
  experiment=CapGraspDG5FS_TriangleXYReach_Z040_PalmClear_ThumbW2_5000ep \
  num_envs=512 \
  max_iterations=5000 \
  early_stop_epochs=9999999999999 \
  dataIndices='[7]' \
  headless=True \
  > "${train_log}" 2>&1
