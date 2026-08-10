#!/usr/bin/env bash
# dg5fs cap-unscrew training.
#
# Task: the cap is a link of the tumbler articulation, held on a screw joint.
# 200 deg of turn raises it 10 mm and the thread lets go; success is getting it
# all the way off while still holding it. The policy drives the wrist (6) and
# every finger joint (20) -- 26 actions.
#
# This replaces train_cap_grasp_curriculum.sh, whose parameters were tuned for a
# different task: a free cap on a cylindrical pedestal, scored on "did it end up
# 30 mm higher", with the wrist driven kinematically. None of CAP_LIFT_*,
# CAP_LEVEL_*, the gravity ramp or the drop thresholds mean anything here. The
# old script is still on disk for the pedestal experiments.
#
# Usage:
#   ./train_cap_unscrew.sh                       # 1000 iters, 512 envs
#   MAX_ITER=5000 NUM_ENVS=512 ./train_cap_unscrew.sh
#   PLAY=1 CKPT=<.pth> ./train_cap_unscrew.sh    # GUI replay
set -euo pipefail

MANIPTRANS_DIR=/home/leegyuwon/Documents/ManipTrans
TOOLS_DIR=/home/leegyuwon/Documents/task1/tools
LOG_DIR=/home/leegyuwon/Documents/task1/logs
CONDA_ENV=/home/leegyuwon/Documents/miniconda3/envs/maniptrans
CONDA_BIN_DIR="${CONDA_ENV}/bin"
CONDA_LIB_DIR="${CONDA_ENV}/lib"
PYTHON_BIN="${CONDA_BIN_DIR}/python"

TAG="${TAG:-Unscrew}"
PLAY="${PLAY:-0}"
MAX_ITER="${MAX_ITER:-1000}"
SAVE_FREQ="${SAVE_FREQ:-100}"
if [[ "${PLAY}" == "1" ]]; then
  NUM_ENVS="${NUM_ENVS:-4}"
else
  NUM_ENVS="${NUM_ENVS:-512}"
fi

mkdir -p "${LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
exp="CapUnscrewDG5FS_${TAG}_${stamp}"
train_log="${LOG_DIR}/cap_unscrew_${TAG}_${stamp}.log"

ckpt_args=()
if [[ -n "${CKPT:-}" ]]; then
  [[ -f "${CKPT}" ]] || { echo "checkpoint not found: ${CKPT}" >&2; exit 2; }
  ckpt_args=(checkpoint="${CKPT}" from_ckpt_epoch=false)
  echo "[unscrew] resuming from ${CKPT}"
fi

mode_args=(headless=True)
redirect=1
if [[ "${PLAY}" == "1" ]]; then
  [[ -n "${CKPT:-}" ]] || { echo "PLAY=1 requires CKPT=<.pth>" >&2; exit 2; }
  exp="CapUnscrewDG5FS_${TAG}_PLAY_${stamp}"
  mode_args=(test=true headless=false)
  redirect=0
  echo "[unscrew] PLAY mode - GUI replay, no training"
fi

echo "[unscrew] ${TAG}  iters=${MAX_ITER}  envs=${NUM_ENVS}"
[[ "${redirect}" == "1" ]] && echo "[unscrew] log=${train_log}"

cd "${MANIPTRANS_DIR}"
[[ "${redirect}" == "1" ]] && exec > "${train_log}" 2>&1

exec env \
  PATH="${CONDA_BIN_DIR}:${PATH}" \
  LD_LIBRARY_PATH="${CONDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
  HYDRA_FULL_ERROR=1 \
  \
  `# --- scene: one tumbler articulation, cap on a screw joint ---` \
  CAP_GRASP_MODE=1 \
  CAP_GRASP_TUMBLER=1 \
  CAP_GRASP_TOOLS_DIR="${TOOLS_DIR}" \
  CAP_GRASP_TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler.urdf \
  CAP_GRASP_LOG="${CAP_LOG:-0}" \
  CAP_DIAG_EVERY="${DIAG_EVERY:-0}" \
  CAP_GRASP_LOG_DIR="${LOG_DIR}" \
  CAP_GRASP_LOG_STAMP="${TAG}_${stamp}" \
  `# 13 = phase 6 + wrist pose in the cap frame (3 pos + 4 quat). The same` \
  `# env var sizes the network target extractor, so both stay in step.` \
  CAP_GRASP_PHASE_DIM=13 \
  CAP_GRASP_CAP_Z="${CAP_Z:-0.225}" \
  CAP_GRASP_EPISODE_STEPS="${EPISODE_STEPS:-360}" \
  CAP_GRASP_RESAMPLE_WRIST_ON_RESET="${RESAMPLE_WRIST:-0}" \
  \
  `# --- action space: wrist + fingers. The fixed-wrist study found 82% of` \
  `#     sampled wrist poses could never succeed on fingers alone, and a` \
  `#     turning couple needs the whole hand anyway. ---` \
  SINGLE_POLICY=1 \
  WRIST_SERVO=0 \
  OBJCENTRIC_OBS=1 \
  \
  `# --- thread: 200 deg -> 10 mm -> released. Must match the asset built by` \
  `#     make_screw_asset.py --pitch 0.018 --max-lift 0.010 ---` \
  CAP_UNSCREW_TARGET_DEG="${UNSCREW_DEG:-200}" \
  `# 12mm on the cap_lift joint (body-relative) = thread 10mm + 2mm carried.` \
  `# Past 10mm the drive is off, so the extra can only be the hand holding it.` \
  `# Does not fully clear the 14.3mm collar; see the separated diagnostic.` \
  CAP_LIFT_SUCCESS_HEIGHT="${LIFT_SUCCESS:-0.012}" \
  CAP_COLLAR_OVERLAP=0.0143 \
  CAP_LIFT_TOL=0.0005 \
  CAP_HOLD_TIME="${HOLD_TIME:-0.3}" \
  CAP_HOLD_DECAY_RATE=3.0 \
  \
  `# --- cap geometry, measured from cap_collision.obj: truncated cone,` \
  `#     r 46.9 mm at the base to 50.0 mm by z=17 mm, 30.0 mm tall.` \
  `#     band_lo is where the tumbler collar ends. Fixed, not tracking the` \
  `#     real exposure as the cap rises: a band that widens with progress` \
  `#     flips the same finger placement from rejected to accepted without` \
  `#     the hand doing anything. ---` \
  CAP_RADIUS_LO=0.0469 \
  CAP_RADIUS_HI=0.0500 \
  CAP_RADIUS_KNEE=0.017 \
  CAP_HEIGHT_M=0.0300 \
  CAP_BAND_LO=0.0143 \
  CAP_FINGER_RADIUS=0.009 \
  CAP_BAND_MARGIN=0.002 \
  CAP_RADIAL_MARGIN=0.010 \
  CAP_PROBE_STEPS=4 \
  CAP_TIP_EXTENSION=0.018 \
  \
  `# --- contact gating. closure_err 0.75 admits >=82 deg of separation;` \
  `#     without it, contacts bunched on one side scored as a grasp. ---` \
  CAP_CONTACT_FORCE_THRESHOLD=0.05 \
  CAP_MIN_CONTACT_TIPS=2 \
  CAP_GRIP_FORCE_REF=0.5 \
  CAP_CLOSURE_ERR_MAX=0.75 \
  \
  `# --- reward. unscrew dominates deliberately: it is the only term that` \
  `#     cannot be collected without a real grasp, so the shaping terms must` \
  `#     not be able to out-earn it. The pedestal run failed exactly there --` \
  `#     reward 757->1755 over 10k epochs with success flat at ~1.8%. ---` \
  CAP_UNSCREW_W="${UNSCREW_W:-10.0}" \
  CAP_HOLD_W="${HOLD_W:-4.0}" \
  CAP_GRASP_CONTACT_W=2.0 \
  CAP_CLOSURE_W="${CLOSURE_W:-3.0}" \
  CAP_GRIP_W=1.0 \
  CAP_THUMB_CONTACT_W=0.5 \
  CAP_TWO_TIP_CONTACT_W=0.5 \
  CAP_NEAR_W=2.0 \
  CAP_TIP_DIST_SCALE=0.06 \
  CAP_NEAR_GATE_ON_GRIP=0.5 \
  \
  `# --- penalties. The wrist is policy-driven now, so it can simply leave;` \
  `#     the leash costs it for drifting off the cap axis. ---` \
  CAP_PENETRATION_PENALTY_W=2.0 \
  CAP_PENETRATION_REF=0.02 \
  CAP_PENETRATION_TOL=0.005 \
  CAP_REVERSE_PENALTY_W=2.0 \
  CAP_REVERSE_REF=0.5 \
  CAP_ACTION_PENALTY_W=0.01 \
  CAP_THUMB_PAIR_W="${THUMB_PAIR_W:-1.0}" \
  CAP_THUMB_PAIR_FACTOR="${THUMB_PAIR_FACTOR:-1.6}" \
  CAP_WRIST_LEASH_W=0.3 \
  CAP_WRIST_LEASH_R=0.16 \
  `# palm-down is enforced by a hard clamp on the wrist orientation` \
  `# (_clamp_unscrew_wrist), not by a penalty. Weight 0 keeps the readout.` \
  CAP_PALM_DOWN_W="${PALM_DOWN_W:-0.0}" \
  CAP_PALM_DOWN_MAX="${PALM_DOWN_MAX:-40}" \
  CAP_PALM_DOWN_REF=40 \
  CAP_WRIST_CLAMP_R=0.25 \
  CAP_WRIST_MAX_LIN_VEL=2.0 \
  CAP_WRIST_MAX_ANG_VEL=10.0 \
  CAP_WRIST_PALM_MAX="${PALM_MAX:-40}" \
  \
  `# --- wrist init: uniform top-down, palm down, just above the cap.` \
  `#     wrist_init.py's mcp_reachable prior is bypassed on purpose. It tests` \
  `#     reachability against a bare r=52mm cylinder standing alone, which on` \
  `#     the tumbler is wrong twice: the collar shrouds the cap's lower 14.3mm` \
  `#     (poses aimed there can never earn contact, since the reward only` \
  `#     counts the 14.3-30.0mm band), and the r=98mm handle is invisible to` \
  `#     it. Still randomised -- just over a simple region, no rejection. ---` \
  CAP_GRASP_WRIST_SAMPLER=topdown \
  CAP_TOPDOWN_R_MAX="${TOPDOWN_R:-0.020}" \
  CAP_TOPDOWN_Z_MIN="${TOPDOWN_Z_MIN:-0.050}" \
  CAP_TOPDOWN_Z_MAX="${TOPDOWN_Z_MAX:-0.065}" \
  CAP_TOPDOWN_TILT_DEG="${TOPDOWN_TILT:-10}" \
  \
  "${PYTHON_BIN}" main/rl/train.py \
  task=ObjDexUnscrew \
  side=RH \
  dexhand=dg5fs \
  experiment="${exp}" \
  num_envs="${NUM_ENVS}" \
  max_iterations="${MAX_ITER}" \
  early_stop_epochs=9999999999999 \
  rl_train.params.config.save_frequency="${SAVE_FREQ}" \
  dataIndices='[7]' \
  `# PhysX default 1000 m/s lets depenetration eject a 0.05 kg cap: measured` \
  `# launches to 19 m/s. 1.0 is the usual value for manipulation.` \
  task.sim.physx.max_depenetration_velocity="${MAX_DEPEN_VEL:-1.0}" \
  "${mode_args[@]}" \
  "${ckpt_args[@]}"
