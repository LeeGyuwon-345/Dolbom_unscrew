#!/usr/bin/env bash
# dg5fs cap-unscrew training, ANYmal-style curriculum variant.
#
# Same task and same scene as train_cap_unscrew.sh -- only the reward
# formulation differs, so the two are directly comparable. This one uses
# cap_unscrew_curriculum_terms:
#
#   * the objective is dense. The other script gates it on a binary
#     grasp_contact (thumb + >=2 contacts + 83 deg opposition), and over 400
#     measured epochs that gate never opened once: opposition stayed at 0%, so
#     the 10.0-weight unscrew term paid exactly zero for the whole run while
#     the policy happily farmed the approach term. Here it is
#     unscrew_frac * (1 - closure_err) * min(contacts/2, 1), which pays
#     partially for a partial grasp and improves toward a real one.
#
#   * constraints ramp in. Following Hwangbo et al. 2019 (arXiv:1901.08652),
#     k_c starts at 0.3 and advances k_c <- k_c^0.997 once per RL iteration,
#     multiplying every constraint term and none of the objective ones. Their
#     reasoning applies directly here: "high penalty on them results in a
#     standing behavior ... such a behavior is already a good local minimum".
#     Our standing behavior is parking the fingers on the cap and not turning.
#
#   * approach counts as objective, not constraint -- in ANYmal velocity
#     tracking is what makes the robot discover walking; here approach is what
#     makes the hand discover touching, so it must not fade early.
#
# Usage:
#   ./train_cap_unscrew_curriculum.sh
#   MAX_ITER=1000 ./train_cap_unscrew_curriculum.sh
#   PLAY=1 CKPT=<.pth> ./train_cap_unscrew_curriculum.sh
#   EXTRA_HYDRA='task.sim.substeps=4' ./train_cap_unscrew_curriculum.sh
set -euo pipefail

MANIPTRANS_DIR=/home/leegyuwon/Documents/ManipTrans
TOOLS_DIR=/home/leegyuwon/Documents/task1/tools
LOG_DIR=/home/leegyuwon/Documents/task1/logs
CONDA_ENV=/home/leegyuwon/Documents/miniconda3/envs/maniptrans
CONDA_BIN_DIR="${CONDA_ENV}/bin"
CONDA_LIB_DIR="${CONDA_ENV}/lib"
PYTHON_BIN="${CONDA_BIN_DIR}/python"

TAG="${TAG:-UnscrewCurr}"
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
exp="CapUnscrewCurrDG5FS_${TAG}_${stamp}"
train_log="${LOG_DIR}/cap_unscrew_curr_${TAG}_${stamp}.log"

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
  exp="CapUnscrewCurrDG5FS_${TAG}_PLAY_${stamp}"
  mode_args=(test=true headless="${HEADLESS:-false}")
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
  `# 손바닥 hull 이 엄지 근위 링크를 삼켜 16524N 유령접촉 -> 엄지 마비.` \
  `# 그 shape 쌍만 필터해서 제외한다 (나머지 자기충돌은 유지).` \
  DEXHAND_SELF_COLLISION_MASK=link_base,link_1_1,link_1_2 \
  CAP_GRASP_TUMBLER=1 \
  CAP_UNSCREW_CURRICULUM=1 \
  `# k0=0.3, kd=0.997 are the paper's values; kc reaches ~0.9 in 400 iters.` \
  CAP_KC0="${KC0:-0.4}" \
  CAP_KD="${KD:-0.997}" \
  `# one advance per RL iteration = horizon_length steps (ObjDexUnscrewPPO)` \
  CAP_KC_PERIOD=32 \
  CAP_GRASP_QUALITY_W="${QUALITY_W:-2.0}" \
  `# 회전 보상은 파지를 0.2초 유지한 뒤에야 지급.` \
  `# quality = (1-closure_err) x 접촉수/CONTACT_REF. 2접촉이면 뒤가 0.5 이므로` \
  `# 0.125 = closure_err<=0.75, 즉 opposed 진단과 같은 83도. grasp_ok(엄지+2접촉)` \
  `# 와 합쳐 "엄지 접촉 AND 2접촉 이상 AND 대향" 이 그대로 stable 조건이 된다.` \
  `# CONTACT_REF 를 바꾸면 이 값도 (2/CONTACT_REF)*0.25 로 같이 바꿔야 한다.` \
  CAP_GRASP_STABLE_MIN="${GRASP_STABLE_MIN:-0.125}" \
  CAP_GRASP_HOLD_TIME="${GRASP_HOLD_TIME:-0.2}" \
  CAP_GRASP_HOLD_W="${GRASP_HOLD_W:-2.0}" \
  CAP_GRASP_TOOLS_DIR="${TOOLS_DIR}" \
  `# GUI 진단용 변형이 같은 폴더에 있다. tumbler_nobodyvis(몸체 안 보임, 물리 동일),` \
  `# tumbler_nobodycol(몸체 충돌 없음). 학습은 항상 기본값으로 돌린다.` \
  CAP_GRASP_TUMBLER_URDF="${TUMBLER_URDF:-/home/leegyuwon/Documents/task1/assets/tumbler/tumbler.urdf}" \
  CAP_GRASP_LOG="${CAP_LOG:-0}" \
  CAP_DIAG_EVERY="${DIAG_EVERY:-0}" \
  CAP_GRASP_LOG_DIR="${LOG_DIR}" \
  CAP_GRASP_LOG_STAMP="${TAG}_${stamp}" \
  `# 13 = phase 6 + wrist pose in the cap frame (3 pos + 4 quat). The same` \
  `# env var sizes the network target extractor, so both stay in step.` \
  CAP_GRASP_PHASE_DIM="${PHASE_DIM:-15}" \
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
  `# --- thread: 180 deg -> 10 mm -> released. Must match the asset built by` \
  `#     make_screw_asset.py --pitch 0.020 --max-lift 0.010 ---` \
  CAP_UNSCREW_TARGET_DEG="${UNSCREW_DEG:-180}" \
  CAP_TURN_ONLY="${TURN_ONLY:-0}" \
  CAP_TURN_PROGRESS="${TURN_PROGRESS:-0}" \
  CAP_UNSCREW_RATE_REF="${UNSCREW_RATE_REF:-2.0}" \
  CAP_UNSCREW_NORM_DEG="${UNSCREW_NORM_DEG:-360}" \
  CAP_HIDE_ANGLE="${HIDE_ANGLE:-0}" \
  `# 나사 끝. URDF 조인트 상한(182도)보다 2도 낮아야 각도가 이 값에 도달한다.` \
  CAP_ENGAGE_DEG="${ENGAGE_DEG:-180}" \
  `# 12mm on the cap_lift joint (body-relative) = thread 10mm + 2mm carried.` \
  `# Past 10mm the drive is off, so the extra can only be the hand holding it.` \
  `# Does not fully clear the 14.3mm collar; see the separated diagnostic.` \
  CAP_LIFT_SUCCESS_HEIGHT="${LIFT_SUCCESS:-0.016}" \
  `# 나사를 끝까지 푼 것 자체에 대한 보상. 마지막 3.5도를 넘을 유인.` \
  CAP_RELEASE_W="${RELEASE_W:-3.0}" \
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
  `# link_X_tip.STL measures 16.1mm across its palmar axis, so the pad sits` \
  `# 8.1mm off the link origin. At 9.0mm dist bottomed out with the finger` \
  `# still ~1mm clear of the wall, where the contact force is exactly zero.` \
  CAP_FINGER_RADIUS="${FINGER_RADIUS:-0.0081}" \
  CAP_BAND_MARGIN=0.002 \
  CAP_RADIAL_MARGIN=0.010 \
  CAP_PROBE_STEPS=4 \
  CAP_TIP_EXTENSION=0.018 \
  \
  `# --- contact gating. closure_err 0.75 admits >=82 deg of separation;` \
  `#     without it, contacts bunched on one side scored as a grasp. ---` \
  CAP_CONTACT_FORCE_THRESHOLD=0.05 \
  `# 게이트는 엄지+2접촉. contact_ref 는 별개로, 몇 개를 "꽉 쥔 것"으로 볼지.` \
  `# 둘이 같은 값이던 동안 contact_frac 이 2개에서 포화해 3~4번째 손가락이` \
  `# 목적항에 전혀 기여하지 못했다.` \
  `# 접촉은 link_X_tip 하나만. link_X_4 까지 세면 깊게 감싸 중위마디로만` \
  `# 누르는 자세가 손끝 파지와 같은 점수를 받는다 (실측 중위 3.36 : 손끝 2.68).` \
  CAP_CONTACT_TIP_ONLY="${TIP_ONLY:-1}" \
  CAP_MIN_CONTACT_TIPS=2 \
  CAP_CONTACT_REF="${CONTACT_REF:-4}" \
  CAP_PALM_FACING_MIN_COS="${PALM_FACING_COS:-0.8}" \
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
  CAP_LIFT_W="${LIFT_W:-4.0}" \
  CAP_LIFT_RELEASE_H=0.010 \
  CAP_TIP_DIST_SCALE=0.06 \
  `# Short-range closure. near at a 60mm scale reads 1.88/2.00 with the fingers` \
  `# still 3.7mm off the wall, so it cannot pay for the last few millimetres;` \
  `# this does, and only when the thumb and one other finger both close.` \
  CAP_PINCH_W="${PINCH_W:-2.0}" \
  CAP_PINCH_DIST_SCALE="${PINCH_SCALE:-0.008}" \
  `# picks up where pinch saturates: same thumb+one-other rule, on force` \
  CAP_PRESS_W="${PRESS_W:-1.0}" \
  `# 손끝 위치 기반 대향 결손. 접촉 없이도 연속이라 첫 스텝부터 기울기가 있다.` \
  CAP_OPPOSE_W="${OPPOSE_W:-4.0}" \
  `# 엄지-검지 방위각이 이 값보다 좁을 때만 부과. 이상이면 0.` \
  CAP_OPPOSE_SEP_DEG="${OPPOSE_SEP:-90}" \
  CAP_NEAR_GATE_ON_GRIP=0.5 \
  \
  `# --- penalties. The wrist is policy-driven now, so it can simply leave;` \
  `#     the leash costs it for drifting off the cap axis. ---` \
  CAP_PENETRATION_PENALTY_W="${PEN_W:-3.0}" \
  CAP_PENETRATION_REF=0.02 \
  CAP_PENETRATION_TOL="${PEN_TOL:-0.002}" \
  CAP_REVERSE_PENALTY_W=2.0 \
  CAP_REVERSE_ANGLE_W="${REVERSE_ANGLE_W:-2.0}" \
  CAP_REVERSE_ANGLE_REF_FRAC=0.25 \
  CAP_REVERSE_REF=0.5 \
  CAP_ACTION_PENALTY_W=0.01 \
  CAP_THUMB_PAIR_W="${THUMB_PAIR_W:-0.0}" \
  CAP_THUMB_PAIR_SEP_DEG="${THUMB_PAIR_SEP:-90}" \
  CAP_WRIST_LEASH_W=0.3 \
  CAP_WRIST_LEASH_R="${LEASH_R:-0.10}" \
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
   \
  CAP_TOPDOWN_Z_FROM_TOP=1 \
  CAP_TOPDOWN_Z_MIN="${TOPDOWN_Z_MIN:-0.020}" \
  CAP_TOPDOWN_Z_MAX="${TOPDOWN_Z_MAX:-0.100}" \
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
  "${ckpt_args[@]}" \
  ${EXTRA_HYDRA:-}
