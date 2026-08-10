# Dolbom Unscrew — RB5-850E + Allegro Hand 텀블러 캡 unscrew RL

Isaac Gym + ManipTrans 기반, RB5-850E 협동로봇 팔 + Allegro Hand(오른손)로
텀블러 캡을 돌려 여는 정책을 학습하는 프로젝트.

## 구조

- `ManipTrans/` — [ManipTrans](https://github.com/ManipTrans/ManipTrans) (GPLv3,
  upstream 커밋 `a3d08cf`) 스냅샷 + 본 프로젝트 수정분:
  - `maniptrans_envs/lib/envs/tasks/dexhandmanip_sh.py` — cap unscrew 환경
    (팔 모드, 저항 랜덤 커리큘럼, 에너지 항, 전력/접촉 계측)
  - `maniptrans_envs/lib/envs/dexhands/rb5_allegro.py` — RB5+Allegro 핸드 등록
  - `main/cfg/task/ObjDexUnscrew.yaml`, `main/cfg/rl_train/ObjDexUnscrewPPO.yaml`
- `task1/` — 프로젝트 본체:
  - `tools2_allegro/` — 학습 런처(`train_cap_unscrew_arm.sh`), 보상 모듈, 포즈
  - `tools/screw_coupling.py` — 나사 구속(mimic) + 저항 구현
  - `assets/` — RB5+Allegro URDF(vmount), 텀블러 자산
  - `grasp_init/` — 초기 파지 최적화

## 데이터 준비 (필수, 저장소 미포함)

데이터셋은 라이선스상 재배포가 불가하여 직접 받아 `ManipTrans/data/`에 배치:

1. [OakInk-v2](https://oakink.net/v2/) — `anno_preview/`, `OakInk-v2-hub/`,
   `OakInk-v2-meta/` (등록 후 다운로드)
2. [SMPL-X / MANO](https://smpl-x.is.tue.mpg.de/) — `body_utils/` 구성용

cap 과제는 trajectory-free지만 초기화 시 데이터셋 로더가 1회 동작한다
(`dataIndices=[7]` 시퀀스 pkl + SMPL-X 모델).

## 학습

```bash
cd task1/tools2_allegro
TAG=RUN1 NUM_ENVS=512 MAX_ITER=10000 \
  ARM_HAND_OBS=0 \
  RB5_ALLEGRO_URDF=$PWD/../assets/rb5_allegro/rb5_allegro_vmount.urdf \
  HAND_FRICTION=4.0 OBJ_FRICTION=2.0 \
  TUMBLER_URDF=$PWD/../assets/tumbler/tumbler_cyl_fric02.urdf \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_RANDOM=1 CAP_RESIST_LO=0.2 \
  CAP_RESIST_START=0.4 CAP_RESIST_END=1.2 \
  CAP_NO_RESET_ON_SUCCESS=1 \
  CAP_SCREW_HOLD_FORCE=2000 TUMBLER_POS=0.4,0,0 \
  CAP_DEMO_FOLLOW_TUMBLER=1 CAP_TUMBLER_RESET_CREATION=1 \
  INIT_POSE=$PWD/poses/arm_grasp_init_closed_vmount.json \
  CAP_RESET_FINGER_TARGETS=1 ACTION_DELTA_SCALE_ARM=0.02 \
  DROP_FAIL_PENALTY=300 EPISODE_STEPS=360 \
  ./train_cap_unscrew_arm.sh
```

재생(GUI)은 `PLAY=1 CKPT=<pth>` 를 추가. 에너지 항은
`CAP_ENERGY_W_HAND/W_ARM/ALPHA` + `CAP_ENERGY_SR_RAMP`(성공률 게이트 램프),
전력 계측은 `CAP_POWER_DEBUG=1`.

주의: 일부 스크립트에 절대경로 기본값이 남아 있어 환경변수로 덮어써야 한다
(위 예시처럼 URDF/INIT_POSE 명시).

## 라이선스

ManipTrans(GPLv3) 파생이므로 본 저장소도 GPLv3을 따른다. `LICENSE` 참조.
