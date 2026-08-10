#!/usr/bin/env bash
# tools2: 두 번째 강화학습. 환경(dexhandmanip_sh)과 씬은 기존과 동일하고,
# 다른 점은 두 가지다.
#
#   * 보상/성공 조건 -- tools2/cap_unscrew_curriculum_terms.py 를 로드한다.
#     CAP_GRASP_TOOLS_DIR 가 이 폴더를 가리키면 환경이 tools 대신 이쪽 모듈을
#     잡고, 여기 없는 공용 모듈(screw_coupling 등)은 tools 로 fallback 한다.
#     지금은 tools 판의 사본이며, 성공 조건과 보상은 재작성 예정.
#
#   * 파지 초기자세 -- grasp_init 최적화가 낸 26차원(손가락 20 + 손목 6)을
#     리셋에 물린다. poses/grasp_26d.json (= grasp_init/poses/pad_area4.json,
#     캡슐 자기충돌/침투/대향까지 만족한 자세). 기본 절반의 환경만 이 자세로
#     시작한다 -- 전부 고정하면 그 자세 전용 정책이 된다.
#
# Usage:
#   ./train_cap_unscrew2.sh
#   MAX_ITER=1000 ./train_cap_unscrew2.sh
#   PLAY=1 CKPT=<.pth> ./train_cap_unscrew2.sh
#   INIT_FRAC=0 ./train_cap_unscrew2.sh          # 초기자세 끄기
set -euo pipefail

MANIPTRANS_DIR=/home/leegyuwon/Documents/ManipTrans
TOOLS_DIR=/home/leegyuwon/Documents/task1/tools2
LOG_DIR=/home/leegyuwon/Documents/task1/logs
CONDA_ENV=/home/leegyuwon/Documents/miniconda3/envs/maniptrans
CONDA_BIN_DIR="${CONDA_ENV}/bin"
CONDA_LIB_DIR="${CONDA_ENV}/lib"
PYTHON_BIN="${CONDA_BIN_DIR}/python"

TAG="${TAG:-Rl2}"
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
exp="CapUnscrew2DG5FS_${TAG}_${stamp}"
train_log="${LOG_DIR}/cap_unscrew2_${TAG}_${stamp}.log"

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
  exp="CapUnscrew2DG5FS_${TAG}_PLAY_${stamp}"
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
  `# --- 파지 초기자세: grasp_init 최적화의 26차원. 절반만 적용(다양성 유지),` \
  `#     관절 노이즈 0.05rad. INIT_FRAC=0 으로 끈다. ---` \
  `# 고정 초기값: grasp_26d 전 관절 +0.03rad (살짝 눌림 -2~-4mm, cos 유지).` \
  `# +0.3/z+10 판은 cos 4/5 미달 + 관통 10~44mm + 밴드 이탈로 실측 폐기.` \
  CAP_GRASP_INIT_POSE="${INIT_POSE:-/home/leegyuwon/Documents/task1/tools2/poses/grasp_26d_p03.json}" \
  CAP_GRASP_INIT_FRAC="${INIT_FRAC:-1.0}" \
  CAP_GRASP_INIT_NOISE="${INIT_NOISE:-0.0}" \
  CAP_GRASP_INIT_WRIST_NOISE_POS="${INIT_WRIST_NOISE_POS:-0.0}" \
  CAP_GRASP_INIT_WRIST_NOISE_ROT="${INIT_WRIST_NOISE_ROT:-0.0}" \
  CAP_GRASP_INIT_POOL="${INIT_POOL:-/home/leegyuwon/Documents/task1/tools2/poses/pool_m05.pt}" \
  `# GUI 진단용 변형이 같은 폴더에 있다. tumbler_nobodyvis(몸체 안 보임, 물리 동일),` \
  `# tumbler_nobodycol(몸체 충돌 없음). 학습은 항상 기본값으로 돌린다.` \
  `# free6: 해제 후 캡이 완전 자유(수평·기울임·계속 회전). 2자유도 판은 z만 가능.` \
  `# 기본 = 원기둥 근사(캡 r50.2/h30, 몸체 R50/H239.4, 64각) + 나사 저항 0.8Nm.` \
  `# 실물 mesh 판은 tumbler_free6.urdf (Main12 재현용).` \
  CAP_GRASP_TUMBLER_URDF="${TUMBLER_URDF:-/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_fric08.urdf}" \
  CAP_GRASP_LOG="${CAP_LOG:-0}" \
  CAP_DIAG_EVERY="${DIAG_EVERY:-0}" \
  CAP_GRASP_LOG_DIR="${LOG_DIR}" \
  CAP_GRASP_LOG_STAMP="${TAG}_${stamp}" \
  `# 13 = phase 6 + wrist pose in the cap frame (3 pos + 4 quat). The same` \
  `# env var sizes the network target extractor, so both stay in step.` \
  `# tools2: 11 = t/T,sin,cos + 해제여부(원값) + 손목 상대 7. 각도류는 차원째` \
  `# 없음. released 는 HIDE_ANGLE 과 무관하게 들어간다.` \
  CAP_GRASP_PHASE_DIM="${PHASE_DIM:-11}" \
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
  `# tools2 는 각도 비노출이 기본. turn(풀림각/180)과 remain(남은각/10도)은` \
  `# 결승선을 관측에 새기는 항이고, 실기에서 누적각은 어차피 관측 불가다.` \
  `# 정책이 보는 회전 정보는 privileged 의 캡 쿼터니언(원시 자세)뿐 -- 목표` \
  `# 180도까지는 감김 모호성이 없다. 성공 판정은 환경이 내부적으로 하므로` \
  `# 정책이 몰라도 채점은 된다.` \
  CAP_HIDE_ANGLE="${HIDE_ANGLE:-1}" \
  `# 상승 속도(캡 z속도) 관측 차단. 위치는 유지 -- 접근에 필요.` \
  CAP_HIDE_RISE="${HIDE_RISE:-1}" \
  `# 성공(해제)해도 에피소드 유지 -- 완주가 수입을 끊지 않게. 성공은 래치 집계.` \
  CAP_NO_RESET_ON_SUCCESS="${NO_RESET_ON_SUCCESS:-1}" \
  `# 손가락 액션 = 전 스텝 실측 관절각 기준 변화량 (rad). 액션 0 = 유지라` \
  `# grasp_init 초기 파지가 랜덤 초기 출력에 깨지지 않는다. 0.1rad/step = 344도/s.` \
  CAP_ACTION_DELTA="${ACTION_DELTA:-1}" \
  CAP_ACTION_DELTA_SCALE="${ACTION_DELTA_SCALE:-0.05}" \
  `# 델타 기준: q(순응, 기준값) | target(실제 서보형 반력). lag = 서보 추종 한계.` \
  CAP_ACTION_DELTA_BASE="${ACTION_DELTA_BASE:-target}" \
  CAP_ACTION_DELTA_LAG="${ACTION_DELTA_LAG:-0.15}" \
  `# 손가락 드라이브 물성. 기본은 기존값(500/30, effort 는 URDF 7.5Nm).` \
  `# DOF_EFFORT 를 주면 URDF 값을 덮어쓴다.` \
  CAP_DOF_STIFFNESS="${DOF_STIFFNESS:-300}" \
  CAP_DOF_DAMPING="${DOF_DAMPING:-18}" \
  CAP_DOF_EFFORT="${DOF_EFFORT:-2.0}" \
  `# 손목도 변화량: 내부 목표를 스텝당 pos 5mm / rot 0.05rad 까지 옮기고 PD 로` \
  `# 추종. 액션 0 = 손목 유지. 랜덤 초기 정책이 파지를 못 깨게 하는 짝.` \
  CAP_WRIST_DELTA="${WRIST_DELTA:-1}" \
  CAP_WRIST_DELTA_POS="${WRIST_DELTA_POS:-0.005}" \
  CAP_WRIST_DELTA_ROT="${WRIST_DELTA_ROT:-0.05}" \
  `# 상승 속도(캡 z속도)도 관측에서 차단. 위치는 유지 -- 접근에 필요.` \
  CAP_HIDE_RISE="${HIDE_RISE:-1}" \
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
  `# 성공 = 해제 + 이후 0.5초 실제 유지 (free6 에서 즉시 떨궈도 성공 방지)` \
  CAP_RELEASED_HOLD_SUCCESS_TIME="${RELEASED_HOLD_SUCCESS_TIME:-0.5}" \
  `# 캡이 홈보다 30mm 아래로 = 낙하 -> 즉시 종료 (바닥 재파지 구멍 차단)` \
  CAP_UNSCREW_DROP_FAIL_Z="${DROP_FAIL_Z:-0.03}" \
  `# 접촉 마찰. PhysX 평균 결합 -> 유효 μ = (손+물체)/2 = 1.5 (실리콘-플라스틱 중심치.` \
  `# 구판 4.0/2.0 = 유효 3.0 은 깨끗한 실리콘 낙관치였음)` \
  CAP_HAND_FRICTION="${HAND_FRICTION:-1.5}" \
  CAP_OBJ_FRICTION="${OBJ_FRICTION:-1.5}" \
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
  `# 방향 가중 접촉 (tools2): cos^2 를 가중치로, 0.8 계단은 진단 정의로만.` \
  CAP_PALM_GRADE="${PALM_GRADE:-1}" \
  `# closure 를 접촉 부채꼴 넓이로 (Main10 실험). 기본 꺼짐 = 기존 대향각.` \
  CAP_CLOSURE_AREA="${CLOSURE_AREA:-1}" \
  CAP_CLOSURE_AREA_REF="${CLOSURE_AREA_REF:-4.0e-3}" \
  CAP_PALM_GRADE_POW="${PALM_GRADE_POW:-2.0}" \
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
  CAP_GRIP_W="${GRIP_W:-1.0}" \
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
  CAP_OPPOSE_W="${OPPOSE_W:-0.0}" \
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
  `# 130mm: 고정 초기 손목이 축에서 102.4mm 라 100mm 기준이면 리셋 첫 스텝부터` \
  `# 최적화 자세를 벌한다. 유출 방지 역할은 130mm 로도 충분.` \
  CAP_WRIST_LEASH_R="${LEASH_R:-0.13}" \
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
