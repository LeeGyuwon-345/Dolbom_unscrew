# A41 정본 세팅 (2026-08-19 사용자 확정)

정직 물리(훈련=재생 일치)에서 J6-롤 90%+ 를 달성한 기준 구성.
런 `CapUnscrewArm_A41_20260818_214632`, **정본 체크포인트 = ep2100**
(신선 검증: 분포 U(0.2,1.2) 해제 99.9% / 코너 1.2 고정 99.8%, 각 ~1,290ep.
ep16700 은 98.5%/97.8%로 온건 퇴화 — 완주 직후 체크포인트가 정본).

## 구성 요약

| 축 | 값 | 근거 |
|---|---|---|
| 관측 | 캡 풀관측 (쿼터니언 간접, 풀림각 직접입력 없음) | A12~A30 계보 |
| 시간 예산 | 360스텝 | 풀관측 템포 (A16c 120스텝 완주) |
| 가속 리미터 | 없음 | A38w 판별 — 배포 최종본에서만 α=10 복원 |
| 물리 | HAND_FRICTION 4.0 / OBJ_FRICTION 6.0 (유효 ~5.0) + randomize=False | 구 숨은 DR 의 명시 재현 (A36/A37 판별) |
| 무중력 램프 | CAP_GRAVITY_RAMP_STEPS=1920 (학습 최초 1920스텝 0→정상) | A39/A40 판별 — 리미터 없는 스크래치의 파지 보호 필수 |
| 저항 | 스칼라 커리큘럼 0.2→1.2 (+0.1, sr게이트 0.8, 창 4096ep) | A35 |
| 보상 | 에너지(sr램프 0.5) + 등급형 풀그립(THUMB_ROT_W 2.0 × cc/4) + 접촉-비례 수입(UNSCREW_CONTACT_SCALE) + NO_RESET + 낙하벌 300 | A19b/A29/A30/A34 |
| 초기자세 | arm_grasp_init_closed_vmount.json + RESET_FINGER_TARGETS | A12 |
| 팔 델타 | 0.02 rad/step | A12 |

## 기동 (전체 env — 이대로 복사)

```bash
env \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=300 EPISODE_STEPS=360 \
  EXTRA_HYDRA=task.task.randomize=False HAND_FRICTION=4.0 OBJ_FRICTION=6.0 \
  INIT_POSE=/home/leegyuwon/Documents/task1/tools2_allegro/poses/arm_grasp_init_closed_vmount.json \
  RB5_ALLEGRO_URDF=/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro_vmount.urdf \
  TUMBLER_POS=0.4,0,0 TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_fric02.urdf \
  CAP_DEMO_FOLLOW_TUMBLER=1 \
  CAP_ENERGY_ALPHA=0.1 CAP_ENERGY_SR_RAMP=0.5 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_W_HAND=1.0 \
  CAP_RESET_FINGER_TARGETS=1 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_START=0.2 CAP_RESIST_END=1.2 CAP_RESIST_STEP=0.1 \
  CAP_RESIST_MIN_EPISODES=4096 CAP_RESIST_SR_GATE=0.8 \
  CAP_ROT_GRIP=1 CAP_SCREW_HOLD_FORCE=2000 CAP_THUMB_ROT_W=2.0 \
  CAP_TUMBLER_RESET_CREATION=1 CAP_UNSCREW_CONTACT_SCALE=1 \
  CAP_GRAVITY_RAMP_STEPS=1920 \
  TAG=<태그> NUM_ENVS=512 MAX_ITER=20000 HEADLESS=1 \
  ./train_cap_unscrew_arm.sh
```

## 재생/검증 프로토콜

- randomize 조작 불필요 (훈련부터 False — 재생과 자동 일치)
- **재생 시 CAP_GRAVITY_RAMP_STEPS 는 넣지 말 것** (학습 전용 부트스트랩 보조)
- 검증은 동결 체크포인트 + 신선 env 그리드: 분포(RESIST_RANDOM=1 LO=0.2
  START/END=1.2) + 코너(1.2 고정), CAP_ANGLE_DUMP=1 로 해제율 집계
- 라이브 sr 은 게이트일 뿐 — 판정은 반드시 신선 그리드로

## 이 정본 위의 후속 카드

① 각도류 관측 이식 (ANGLE_ONLY+ANGLE_VEL, lr=1e-5 연착륙, A23b 공식 —
A41 은 저항 커리큘럼 이력이 있어 A24 조건 충족) ② 시간 압축 360→240
③ 가속 리미터 α=10 복원 fine-tune (배포 최종본) ④ 마찰 하강 5→3→1.5
⑤ 초기자세 노이즈 → export 패키지

## ep2100 접촉 실측 (저항 적응 파지)

회전/유지 접촉: 0.2Nm 1.85/2.79 (경제적 얇은 파지) ↔ 1.2Nm 3.16/3.80
(A30 2.8/3.8, A16c 3.27 대비 우수). 회전 중 손가락 1개 순환 이탈은
계보 공통 정상 패턴. 유의: 저저항의 얇은 파지는 세계-고정 텀블러 덕일
수 있음 — 실기 이식에서 미덥지 않으면 접촉 하한 강화(CONTACT_REF/등급
지수) 카드.
