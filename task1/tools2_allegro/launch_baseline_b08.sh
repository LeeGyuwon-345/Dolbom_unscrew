#!/bin/bash
# ===== 은닉 정본 (2026-09-03 승격, 원본 run B08_20260902_231948) =====
# 실기 배포 계열 기준. 비전 불요, actor 는 캡 상태를 일절 안 본다.
#
# B08 = 이전 정본(B02) + 실측 반영 기구/배치 4 + 검증된 2변수.
#
#  [기구·배치 — 실물 실측 반영, 성능 실험 아님]
#    텀블러 지그   +31mm -> +22mm            TUMBLER_POS z 0.031 -> 0.022
#    텀블러 몸체   215mm -> 228mm            tumbler_cyl_r44_fric02_H228.urdf
#    캡 원점       0.246 -> 0.250            = 0.022 + 0.228 (월드 절대 z)
#    마운트 기구   hand_mount y -0.060 -> -0.0615
#                  RB5 마운트면-손등 30.5mm -> 32.0mm. 파일 직접 수정이라 env 에 안 나타남.
#                  옛 기구 사본: rb5_allegro_vmount_siltip2_y060.urdf
#    초기자세      up31 -> rhinit_z193_H228.json (손목 +4mm, 손 16축 불변)
#
#  [실험 변수 2 — B03~B07 에서 검증하고 남긴 것]
#    CAP_RESIST_STEP=0.3        저항 계단 0.1->0.3 (35계단 -> 13계단)
#    CAP_TUMBLER_POS_NOISE_MM=2 지그 오차 ±2mm (관측 기준은 공칭 고정 = 실기 상황과 동일)
#
#  보상·관측은 B02 정본과 **완전히 동일**하다(GRIP_FLOOR 4,2,4,4 / SLIP2_W 5.0 / CONTACT_REF 4 …).
#
# 승격 근거 (2026-09-03 실측):
#   - 형성 **재추첨 0회**(B02 는 1회). 지그 노이즈가 형성을 막지 않음을 확인
#   - 저항 커리큘럼 13/13 완주, 구간 내내 sr 0.96~0.99.
#     4.0N·m 도달 ep2400 대 — 정본(ep4032) 대비 **1,600 epoch 단축**
#   - 마찰 2.0 도달 ep2900
#   - **ep2800 @저항4.0N·m: 캡 180° · J6 기여 90%(161°) · 압착률 엄지100/검지100/중지81/약지100%**
#     => J롤과 4지 접촉을 **동시에** 얻은 첫 사례. B02 는 J롤은 되나 검지27%/약지10%,
#        B04~B07 은 4지는 되나 J롤이 죽었다(J6 17~45%).
#   - z 슬립 4.8mm (B02 6.5mm 보다 작음)
#
# ⚠️ 한계: 원본 run 은 ep3200(마찰 전이 회복 중, sr 0.45)에서 사용자 지시로 중지했다.
#   20000ep 완주본이 아니다. 배포 후보는 회복 후 체크포인트로 갱신할 것.
#   마찰 2.0 전이 붕괴(ep2900 sr 0.185)는 B02 와 같은 정상 경로다.
#
# TAG 자동 생성(B08B_). setsid nohup 으로 기동할 것.
exec env CAP_TUMBLER_POS_NOISE_MM=2 TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_r44_fric02_H228.urdf CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.250 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=4,2,4,4 CAP_GRIP_FLOOR_RAMP=8.0 GRIP_W=2.0 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B08B_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_STEP=0.3 CAP_RESIST_START=0.2 CAP_RESIST_END=4.0 \
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
