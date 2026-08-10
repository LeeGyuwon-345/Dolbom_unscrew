#!/usr/bin/env bash
set -euo pipefail

CURRENT_PID="${CURRENT_PID:-74106}"
CURRENT_RUN_DIR="${CURRENT_RUN_DIR:-/home/leegyuwon/Documents/ManipTrans/runs/CapGraspDG5FS_ThumbReach__07-28-17-48-12}"
CURRENT_LOG="${CURRENT_LOG:-/home/leegyuwon/Documents/task1/logs/cap_grasp_train_20260728_174809.log}"
LOG_DIR="${LOG_DIR:-/home/leegyuwon/Documents/task1/logs}"
MANIPTRANS_DIR="${MANIPTRANS_DIR:-/home/leegyuwon/Documents/ManipTrans}"
PYTHON_BIN="${PYTHON_BIN:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin}"
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib}"

echo "[watch] waiting for current training pid=${CURRENT_PID}"
echo "[watch] current_run=${CURRENT_RUN_DIR}"
echo "[watch] current_log=${CURRENT_LOG}"

while kill -0 "${CURRENT_PID}" 2>/dev/null; do
  sleep 60
done

echo "[watch] current training process exited at $(date '+%F %T')"

ckpt=""
for _ in $(seq 1 20); do
  ckpt="$(
    find "${CURRENT_RUN_DIR}/nn" -maxdepth 1 -type f \
      -name 'last_CapGraspDG5FS_ThumbReach_ep_3000*.pth' \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -n "${ckpt}" ]]; then
    break
  fi
  sleep 30
done

if [[ -z "${ckpt}" ]]; then
  echo "[watch] no ep_3000 last checkpoint found. Not starting continuation."
  echo "[watch] last current log lines:"
  tail -n 80 "${CURRENT_LOG}" || true
  exit 2
fi

stamp="$(date +%Y%m%d_%H%M%S)"
train_log="${LOG_DIR}/cap_grasp_train_continue10k_${stamp}.log"

echo "[watch] found checkpoint=${ckpt}"
echo "[watch] starting continuation at $(date '+%F %T')"
echo "[watch] continuation_log=${train_log}"

cd "${MANIPTRANS_DIR}"
exec env \
  PATH="${CONDA_BIN_DIR}:${PATH}" \
  LD_LIBRARY_PATH="${CONDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
  HYDRA_FULL_ERROR=1 \
  CAP_GRASP_MODE=1 \
  CAP_GRASP_TOOLS_DIR=/home/leegyuwon/Documents/task1/tools \
  CAP_GRASP_URDF=/home/leegyuwon/Documents/task1/assets/cap_only/cap_only.urdf \
  CAP_GRASP_LOG_STAMP="continue10k_${stamp}" \
  CAP_GRASP_LOG_DIR="${LOG_DIR}" \
  CAP_GRASP_WRIST_SAMPLER=mcp_reachable \
  CAP_REACH_CONTACT_MARGIN=0.008 \
  CAP_REACH_MIN_FINGERS=2 \
  CAP_REACH_REQUIRE_THUMB=1 \
  CAP_THUMB_REACH_MARGIN=0.020 \
  CAP_THUMB_REACH_OPPOSITION_STEPS=7 \
  CAP_THUMB_REACH_FLEX_STEPS=11 \
  CAP_THUMB_REACH_CURL_STEPS=3 \
  CAP_REACH_MCP_STEPS=17 \
  CAP_REACH_SEGMENT_SAMPLES=5 \
  CAP_GRASP_WRIST_MAX_TRIES=500 \
  CAP_GRASP_TIME=1.0 \
  CAP_SETTLE_TIME=0.5 \
  CAP_HOLD_TIME=2.0 \
  CAP_GRAVITY_RAMP_TIME=0.0 \
  "${PYTHON_BIN}" main/rl/train.py \
  task=ObjDexUnscrew \
  side=RH \
  dexhand=dg5fs \
  experiment=CapGraspDG5FS_ThumbReach_Cont10k \
  num_envs=512 \
  max_iterations=10000 \
  early_stop_epochs=9999999999999 \
  dataIndices='[7]' \
  headless=True \
  checkpoint="${ckpt}" \
  from_ckpt_epoch=false \
  > "${train_log}" 2>&1
