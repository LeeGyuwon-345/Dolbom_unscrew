#!/usr/bin/env bash
# A37 재생 전용 래퍼 — 학습 프로세스(/proc environ 실측)와 동일 env 보장.
#
# 변수는 두 부류이고 전달 방식이 다르다:
#  1) 런처 관리형: 런처가 CAP_X="${X:-기본값}" 로 되쓰므로 반드시
#     "입력변수"(bare 이름)로 전달 (INIT_POSE, TUMBLER_*, EPISODE_STEPS 등).
#  2) 통과형: 런처가 건드리지 않는 CAP_* — 래퍼가 직접 export 한 것이라
#     CAP_ 이름 그대로 전달 (RESIST 커리큘럼, 가속 리미터, 각도-온리 관측,
#     DEMO_FOLLOW/RESET_CREATION, 에너지 계수 등 20개).
# 이 구분을 어기면 재생 env 가 조용히 어긋난다 (2026-08-18 GUI 사고 2건).
#
# 사용법:
#   ./play_a37.sh <CKPT.pth>                # GUI (기본 1 env)
#   HEADLESS=true NUM_ENVS=64 ./play_a37.sh <CKPT.pth>   # 헤드리스 프로브
set -euo pipefail
CKPT="${1:?사용법: ./play_a37.sh <checkpoint.pth>}"

exec env \
  `# --- 1) 런처 관리형: 입력변수로 전달 ---` \
  ACTION_DELTA_SCALE_ARM=0.02 \
  ARM_HAND_OBS=0 \
  DROP_FAIL_PENALTY=300 \
  EPISODE_STEPS=360 \
  EXTRA_HYDRA=task.task.randomize=False \
  HAND_FRICTION=4.0 \
  OBJ_FRICTION=6.0 \
  INIT_POSE=/home/leegyuwon/Documents/task1/tools2_allegro/poses/arm_grasp_init_closed_vmount.json \
  RB5_ALLEGRO_URDF=/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro_vmount.urdf \
  TUMBLER_POS=0.4,0,0 \
  TUMBLER_URDF="${TUMBLER_URDF:-/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_fric02.urdf}" \
  `# --- 2) 통과형 CAP_*: 그대로 전달 (학습 environ 실측 20개) ---` \
  CAP_ACCEL_LIMIT_ARM=10 \
  CAP_ACCEL_LIMIT_HAND=10 \
  CAP_DEMO_FOLLOW_TUMBLER=1 \
  CAP_ENERGY_ALPHA=0.1 \
  CAP_ENERGY_SR_RAMP=0.5 \
  CAP_ENERGY_W_ARM=1.667 \
  CAP_ENERGY_W_HAND=1.0 \
  CAP_OBS_ANGLE_ONLY=1 \
  CAP_RESET_FINGER_TARGETS=1 \
  CAP_RESIST_CURRICULUM=1 \
  CAP_RESIST_END=1.2 \
  CAP_RESIST_MIN_EPISODES=4096 \
  CAP_RESIST_SR_GATE=0.8 \
  CAP_RESIST_START=0.2 \
  CAP_RESIST_STEP=0.1 \
  CAP_ROT_GRIP=1 \
  CAP_SCREW_HOLD_FORCE=2000 \
  CAP_THUMB_ROT_W=2.0 \
  CAP_TUMBLER_RESET_CREATION=1 \
  CAP_UNSCREW_CONTACT_SCALE=1 \
  `# --- 재생 모드 ---` \
  PLAY=1 \
  HEADLESS="${HEADLESS:-false}" \
  NUM_ENVS="${NUM_ENVS:-1}" \
  MAX_ITER="${MAX_ITER:-2000}" \
  TAG="${TAG:-A37PLAY}" \
  CKPT="${CKPT}" \
  /home/leegyuwon/Documents/task1/tools2_allegro/train_cap_unscrew_arm.sh
