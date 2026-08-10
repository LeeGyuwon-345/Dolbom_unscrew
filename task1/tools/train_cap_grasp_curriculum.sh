#!/usr/bin/env bash
# Staged curriculum for the cap-only dg5fs grasp policy.
#
#   STAGE=0  cap is permanently pinned, gravity never turns on.
#            Goal: learn to make thumb + 2-finger side contact at all.
#            Success = 0.5 s of sustained contact. Wrist poses are fixed per env.
#   STAGE=1  gravity ramps in over 0.5 s, loose drop tolerance, short hold.
#            Goal: learn to actually carry the load.
#   STAGE=2  full task: tight drop tolerance, 2 s hold, wrist resampled each reset.
#
# Each stage resumes from the previous stage's checkpoint:
#   STAGE=0 ./train_cap_grasp_curriculum.sh
#   STAGE=1 CKPT=<stage0 .pth> ./train_cap_grasp_curriculum.sh
#   STAGE=2 CKPT=<stage1 .pth> ./train_cap_grasp_curriculum.sh
#
# NOTE: the observation now carries a 6-dim phase block, so checkpoints from
# before that change are NOT loadable. Start STAGE=0 from scratch.
set -euo pipefail

#
# PLAY=1 replays a checkpoint in the GUI using the *same* stage env config, so
# the viewer can never drift from what the policy was trained on (a separate
# viewer script silently breaks the moment an env var changes -- e.g. omitting
# CAP_GRASP_PHASE_DIM changes the observation width and the net fails to load):
#   PLAY=1 STAGE=0 CKPT=<.pth> ./train_cap_grasp_curriculum.sh
STAGE="${STAGE:-0}"
CKPT="${CKPT:-}"
PLAY="${PLAY:-0}"
# Periodic checkpoint interval in epochs (rl_games also saves a "best" model
# every time mean reward improves, from epoch save_best_after onward).
SAVE_FREQ="${SAVE_FREQ:-200}"
if [[ "${PLAY}" == "1" ]]; then
  NUM_ENVS="${NUM_ENVS:-4}"
else
  NUM_ENVS="${NUM_ENVS:-512}"
fi
LOG_DIR="${LOG_DIR:-/home/leegyuwon/Documents/task1/logs}"
MANIPTRANS_DIR="${MANIPTRANS_DIR:-/home/leegyuwon/Documents/ManipTrans}"
TOOLS_DIR="${TOOLS_DIR:-/home/leegyuwon/Documents/task1/tools}"
PYTHON_BIN="${PYTHON_BIN:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin}"
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib}"

case "${STAGE}" in
  0)
    TAG="Stage0Contact"
    MAX_ITER="${MAX_ITER:-1500}"
    # grasp_time far beyond the episode => gravity_scale stays 0 for the whole
    # rollout. The cap is a static body (CAP_GRASP_FIX_CAP), so it cannot be
    # pushed away and no pinning force is needed.
    STAGE_ENV=(
      CAP_GRASP_TIME=1000.0
      CAP_SETTLE_TIME=0.0
      CAP_HOLD_TIME=0.5
      CAP_GRAVITY_RAMP_TIME=0.0
      CAP_REQUIRE_GRAVITY_FOR_HOLD=0
      CAP_GRASP_EPISODE_STEPS=180
      CAP_DROP_THRESHOLD=1.0
      CAP_GRASP_RESAMPLE_WRIST_ON_RESET=0
      CAP_DROP_PENALTY_W=0.0
      CAP_CAP_VEL_PENALTY_W=0.0
      CAP_HOLD_W=2.0
      # Static cap: contacts resolve properly, so no pinning hack is needed.
      CAP_GRASP_FIX_CAP=1
      CAP_GRASP_LOCK_CAP_BEFORE_GRAVITY=0
    )
    ;;
  1)
    TAG="Stage1Gravity"
    MAX_ITER="${MAX_ITER:-2000}"
    STAGE_ENV=(
      CAP_GRASP_TIME=1.0
      CAP_SETTLE_TIME=0.5
      CAP_HOLD_TIME=0.5
      CAP_GRAVITY_RAMP_TIME=0.5
      CAP_REQUIRE_GRAVITY_FOR_HOLD=1
      CAP_GRASP_EPISODE_STEPS=210
      CAP_DROP_THRESHOLD=0.040
      CAP_GRASP_RESAMPLE_WRIST_ON_RESET=0
      CAP_GRASP_FIX_CAP=0
      CAP_GRASP_LOCK_CAP_BEFORE_GRAVITY=1
    )
    ;;
  2)
    TAG="Stage2Full"
    MAX_ITER="${MAX_ITER:-5000}"
    STAGE_ENV=(
      CAP_GRASP_TIME=1.0
      CAP_SETTLE_TIME=0.5
      CAP_HOLD_TIME=2.0
      CAP_GRAVITY_RAMP_TIME=0.3
      CAP_REQUIRE_GRAVITY_FOR_HOLD=1
      CAP_GRASP_EPISODE_STEPS=270
      CAP_DROP_THRESHOLD=0.015
      CAP_GRASP_RESAMPLE_WRIST_ON_RESET=1
      CAP_GRASP_FIX_CAP=0
      CAP_GRASP_LOCK_CAP_BEFORE_GRAVITY=1
    )
    ;;
  P)
    # Pedestal / lift formulation. The cap rests on a static pedestal under
    # normal gravity from t=0 and the wrist is commanded upward at 1.5 s;
    # success = the cap rose >=3 cm with the hand still gripping it.
    # No pin, no gravity schedule, no drop termination -- a failed grasp just
    # leaves the cap sitting there and the episode continues, so the policy
    # gets many attempts per rollout instead of dying at the first slip.
    TAG="StagePLift"
    MAX_ITER="${MAX_ITER:-2000}"
    STAGE_ENV=(
      CAP_GRASP_PEDESTAL=1
      CAP_LIFT_START=1.5
      CAP_LIFT_TIME=1.0
      CAP_LIFT_HEIGHT=0.060
      CAP_LIFT_SUCCESS_HEIGHT=0.030
      CAP_LIFT_W=4.0
      CAP_HOLD_TIME=0.3
      CAP_GRASP_EPISODE_STEPS=240
      CAP_GRASP_RESAMPLE_WRIST_ON_RESET=0
      CAP_GRASP_FIX_CAP=0
      CAP_GRASP_LOCK_CAP_BEFORE_GRAVITY=0
      CAP_CAP_VEL_PENALTY_W=0.0
      # Wrist is driven kinematically: exactly +6 cm from the initial pose,
      # unaffected by contact loads. The policy controls the fingers only.
      CAP_GRASP_LOCK_WRIST=1
    )
    ;;
  *)
    echo "STAGE must be 0, 1, 2 or P (got '${STAGE}')" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
exp="CapGraspDG5FS_${TAG}_${stamp}"
train_log="${LOG_DIR}/cap_grasp_${TAG}_${stamp}.log"

ckpt_args=()
if [[ -n "${CKPT}" ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "checkpoint not found: ${CKPT}" >&2
    exit 2
  fi
  ckpt_args=(checkpoint="${CKPT}" from_ckpt_epoch=false)
  echo "[stage${STAGE}] resuming from ${CKPT}"
fi

mode_args=(headless=True)
redirect=1
if [[ "${PLAY}" == "1" ]]; then
  if [[ -z "${CKPT}" ]]; then
    echo "PLAY=1 requires CKPT=<path to .pth>" >&2
    exit 2
  fi
  exp="CapGraspDG5FS_${TAG}_PLAY_${stamp}"
  mode_args=(test=true headless=false)
  redirect=0
  echo "[stage${STAGE}] PLAY mode - GUI replay, no training"
fi

echo "[stage${STAGE}] ${TAG}  iters=${MAX_ITER}  envs=${NUM_ENVS}"
if [[ "${redirect}" == "1" ]]; then
  echo "[stage${STAGE}] log=${train_log}"
fi

cd "${MANIPTRANS_DIR}"
if [[ "${redirect}" == "1" ]]; then
  exec > "${train_log}" 2>&1
fi
exec env \
  PATH="${CONDA_BIN_DIR}:${PATH}" \
  LD_LIBRARY_PATH="${CONDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
  HYDRA_FULL_ERROR=1 \
  CAP_GRASP_MODE=1 \
  CAP_GRASP_TOOLS_DIR="${TOOLS_DIR}" \
  CAP_GRASP_URDF=/home/leegyuwon/Documents/task1/assets/cap_only/cap_only.urdf \
  CAP_GRASP_LOG_DIR="${LOG_DIR}" \
  CAP_GRASP_LOG_STAMP="${TAG}_${stamp}" \
  CAP_GRASP_PHASE_DIM=6 \
  "${STAGE_ENV[@]}" \
  `# --- wrist init prior (unchanged across stages) ---` \
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
  `# Graspability, not just reachability: thumb and at least one finger must be` \
  `# able to contact the rim >=90 deg apart about the cap axis. Without this,` \
  `# Stage-0 measured closure_err ~= 0.97 (all contacts on one side).` \
  CAP_OPPOSITION_MIN_DEG="${OPPOSITION_MIN_DEG:-90}" \
  CAP_REACH_MCP_STEPS=17 \
  CAP_REACH_SEGMENT_SAMPLES=5 \
  CAP_GRASP_WRIST_MAX_TRIES=3000 \
  `# Pre-gravity pin. Stage 0 does not need this (static cap); stages 1-2 use` \
  `# teleport. A stiff servo was tried and is numerically unstable for a` \
  `# 0.05 kg body at dt=1/60 -- it flings the cap on release (cap_speed 6.1 m/s` \
  `# vs 0.98, reward -219 vs +181). See CAP_CAP_LOCK_POS_KP in dexhandmanip_sh.` \
  CAP_GRASP_CAP_LOCK_MODE="${CAP_LOCK_MODE:-teleport}" \
  `# --- reward shaping (see cap_grasp_rl_terms.CapGraspRLConfig) ---` \
  CAP_FINGER_RADIUS=0.009 \
  CAP_PROBE_STEPS=4 \
  CAP_TIP_EXTENSION=0.018 \
  CAP_NEAR_W=0.5 \
  CAP_NEAR_GATE_ON_GRIP=0.5 \
  CAP_GRIP_W=1.5 \
  CAP_GRIP_FORCE_REF=0.5 \
  CAP_THUMB_CONTACT_W=2.0 \
  CAP_CLOSURE_W=1.0 \
  CAP_PENETRATION_PENALTY_W=2.0 \
  CAP_PENETRATION_TOL=0.005 \
  CAP_HOLD_DECAY_RATE=3.0 \
  CAP_LIFT_REQUIRES_GRASP="${LIFT_REQUIRES_GRASP:-1}" \
  CAP_GRASP_TUMBLER_BODY="${TUMBLER_BODY:-0}" \
  CAP_GRASP_TUMBLER="${TUMBLER:-0}" \
  "${PYTHON_BIN}" main/rl/train.py \
  task=ObjDexUnscrew \
  side=RH \
  dexhand=dg5fs \
  experiment="${exp}" \
  num_envs="${NUM_ENVS}" \
  max_iterations="${MAX_ITER}" \
  early_stop_epochs=9999999999999 \
  `# PhysX default 1000 m/s lets a 0.05 kg cap be ejected by depenetration:`\
  `# measured launches to 19 m/s and over a metre. 1.0 is the usual value`\
  `# for manipulation. Overridden here rather than in the shared task yaml.`\
  task.sim.physx.max_depenetration_velocity="${MAX_DEPEN_VEL:-1.0}" \
  rl_train.params.config.save_frequency="${SAVE_FREQ}" \
  dataIndices='[7]' \
  "${mode_args[@]}" \
  "${ckpt_args[@]}"
