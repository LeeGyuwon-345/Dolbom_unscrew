#!/bin/bash
# ===== 은닉 정본 (신 기구·신 텀블러판) — 2026-09-02 =====
# 보상·커리큘럼·관측은 launch_baseline_b02.sh 와 **완전히 동일**하다.
# 바뀐 것은 실물 실측을 반영한 기구/배치 4가지뿐이다(성능 실험 아님, 기준 이동):
#
#   1) 텀블러 지그 높이   +31mm -> +22mm      TUMBLER_POS z 0.031 -> 0.022
#   2) 텀블러 몸체 높이   215mm -> 228mm      자산 tumbler_cyl_r44_fric02_H228.urdf
#        (make_cyl_tumbler.py --r .044 --h .016 --R .044 --H 0.228 --capz 0.228 --spin-friction 0.2)
#   3) 캡 원점(cap_home)  0.246 -> 0.250      = 텀블러루트 0.022 + 몸체 0.228 (월드 절대 z)
#   4) 마운트 기구        rb5_allegro_vmount_siltip2.urdf 의 hand_mount y -0.060 -> -0.0615
#        RB5 마운트면-손등 최소거리 30.5mm -> **32.0mm** (실측 정합). 파일명이 같으므로
#        이 런처의 env 에는 안 나타난다. 옛 기구 사본: rb5_allegro_vmount_siltip2_y060.urdf
#
#   초기자세: poses/rhinit_z193_H228.json
#        up31 대비 손목만 +4mm (캡이 0.246->0.250 로 4mm 올라간 만큼). 손 16축 불변.
#        검증: 캡 대비 상대 기하 동일(캡벽 갭 검지-2.6 엄지-4.3 중지+2.8 약지-3.6mm).
#
# 원본 정본(31mm/215mm/30.5mm) 재현이 필요하면 launch_baseline_b02.sh 를 그대로 쓸 것.
# TAG 자동 생성(B02G_). setsid nohup 으로 기동할 것.
exec env TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_r44_fric02_H228.urdf CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.250 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=4,2,4,4 CAP_GRIP_FLOOR_RAMP=8.0 GRIP_W=2.0 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B02G_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_START=0.2 CAP_RESIST_END=4.0 \
  INIT_POSE=/home/leegyuwon/Documents/task1/tools2_allegro/poses/rhinit_z193_H228.json \
  RB5_ALLEGRO_URDF=/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro_vmount_siltip2.urdf \
  CAP_DEMO_FOLLOW_TUMBLER=1 CAP_TUMBLER_RESET_CREATION=1 CAP_RESET_FINGER_TARGETS=1 \
  CAP_ROT_GRIP=1 CAP_SCREW_HOLD_FORCE=2000 CAP_THUMB_ROT_W=2.0 CAP_UNSCREW_CONTACT_SCALE=1 \
  CAP_UNSCREW_THUMB_SCALE=1 CAP_THUMB_SCALE_REF=4.0 CAP_REWARD_ZONE_FORCE=1 \
  CAP_NO_RESET_ON_SUCCESS=1 CAP_OBS_HIDE_POS=1 CAP_OBS_HIDE_LINVEL=1 \
  CAP_OBS_TIP_FORCE6=1 CAP_TIP_FORCE_FILTER=0.3 \
  CAP_PROBE_BACKSPAN=0.024 PALM_FACING_COS=0.87 \
  CAP_ACCEL_LIMIT_HAND=50 CAP_ACCEL_LIMIT_ARM=50 \
  /home/leegyuwon/Documents/task1/tools2_allegro/train_cap_unscrew_arm.sh
