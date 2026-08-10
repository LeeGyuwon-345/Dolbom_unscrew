# tools2 기준 구성 (BASELINE)

이 파일이 "현재 학습 방식"의 정의다. `./train_cap_unscrew2.sh` 를 **인자 없이**
실행하면 정확히 이 구성으로 돈다 — 런처의 기본값이 곧 기준이고, 이 문서는
그 기본값들의 목록과 근거다.

## 코드를 고칠 때의 규칙

1. **새 기능은 런처 기본값에 기준을 반영한다.** 환경 코드(dexhandmanip_sh)의
   `os.environ.get(..., "0")` 기본은 tools(구) 호환을 위해 꺼짐으로 두고,
   tools2 런처에서 켠다. 환경 코드 기본값을 바꾸면 구 체크포인트 재생이
   깨진다.
2. **런처 기본값을 바꾸면 이 파일을 같이 갱신한다.** 둘이 어긋나면 이 파일이
   아니라 런처가 실제 동작이다 — 어긋남을 발견하면 그게 버그다.
3. 실험적 변경은 환경변수 오버라이드로 하고(`PALM_GRADE=0 ./train...`),
   결과가 채택될 때만 기본값을 바꾼다.

## 기준 구성 (Main12, 2026-08-06 기준)

### 과제 정의
    성공          캡 해제 (풀림각 180도) + 이후 0.5초 실제 유지
                  (CAP_RELEASED_HOLD_SUCCESS_TIME=0.5, 2026-08-05 추가 --
                   유지 판정 = 해제됨 AND grasp_quality >= stable_min 연속.
                   free6 물리에서 해제 즉시 떨어뜨리는 정책의 성공 집계 방지.
                   에피소드 내 래치는 유지: 0.5초 채운 뒤 떨어뜨려도 성공)
    에피소드      성공해도 끝나지 않음 (CAP_NO_RESET_ON_SUCCESS=1) — timeout 360스텝
    나사          180도 = 10mm 상승 = 해제. 조인트 상한 182도

### 관측 (341차원)
    proprioception 73   관절 q/cos/sin + 손목 base_state (위치 마스킹)
    privileged 57       dq, 캡 위치/자세/속도(z속도 마스킹: CAP_HIDE_RISE=1),
                        tip_force, 무게중심, 무게
    target 211          모방블록 46 + obj2joints 26 + BPS 128 + cap_phase 11
    cap_phase 11        [t/T, sin, cos, released(원값), 손목상대 위치3+자세4]
                        (CAP_GRASP_PHASE_DIM=11, CAP_HIDE_ANGLE=1)
    각도류(turn/remain/hold_frac/lift_frac) 는 차원째 없음

### 액션
    손가락 20      서보 목표 + action×0.05rad (CAP_ACTION_DELTA=1, SCALE=0.05,
                   BASE=target — 명령 적분형, 실기 서보와 동일. lag 클램프 0.15rad)
                   드라이브 300/18/2.0Nm (실기 스톨 토크 2Nm 반영)
    손목 6         내부 목표를 스텝당 ±5mm/±0.05rad 이동, PD 추종
                   (CAP_WRIST_DELTA=1, Kp 100/1.0, Kd 4/0.02, err_max 30mm,
                    힘 20N / 토크 1.5Nm 클램프)
    양쪽 다 EMA 없음. 액션 0 = 현재 유지

### 초기값 (리셋)
    노이즈 풀 poses/pool_m05.pt (INIT_POOL 기본값) — 매 리셋 무작위 인덱싱
    = m05(최적화 파지 −0.05rad, 살짝 편 시작) + 관절 0.04rad/손목 5mm·1.7° 노이즈,
      침투>10mm·자기충돌>3mm 거부(합격률 94%), 10만 개
    풀이 지정되면 INIT_POSE(p03)/INIT_NOISE 는 무시된다.
    (고정 p03 예압 판은 Main5~11 이 사용 — 자세 과적합 때문에 풀로 대체)

### 보상 (tools2/cap_unscrew_curriculum_terms.py)
    목적항 (kc 무관)
      unscrew 10    max(frac, released) × quality × grasp_hold × grasp_ok_f
                    -- 해제 래치: 해제 후 되감겨도 지급 최대치 고정
      hold 4        released × quality × grasp_hold   -- 해제 후 유지
      near 2        exp(-거리/60mm) × 방향, grip 이 크면 감쇠
      pinch 2       엄지+최근접 근거리 폐합
      (hold원판/lift/release/press 는 목적에서 제거, 진단 계산만 유지)
    접촉 판정
      touch = 힘>0.05N AND 밴드(14.3~30mm)          -- 계단 (물리 사실)
      기여도 = touch × clamp(cos,0,1)²               -- 방향은 가중 (PALM_GRADE=1)
      grasp_ok_f = thumb_w × clamp(count/2,0,1)      -- 연속 (AND 절벽 없음)
      0.8 계단은 진단 "인정 접촉" 정의로만 남음
    폐합 (CLOSURE_AREA=1, 기본)
      엄지 부채꼴 삼각형 넓이 — 비엄지 모든 짝 6개, 접촉 가중(w_t×w_a×w_b),
      ref 4.0e-3 m². oppose(엄지-중지 위치 처방)는 OPPOSE_W=0 으로 제거
    제약항 (×kc, kc: 0.4 → 1.0)
      quality 2, ghold 2, grip 1, pen 3,
      reverse 2, revang 2, leash 0.3 (반경 0.13m), action 0.01

### 유인 구조의 핵심 (왜 이 조합인가)
    174도 주차  = 9.7×q/step  (나사산이 각도 보장, 위험 없음)
    해제+유지   = 10×q + 4×q ≈ 14×q/step, 되감김과 무관 (래치)
    → 해제가 모든 경우에 주차를 +44% 이김. Main3 에서 래치 없이
      ep300 17.2% → ep600 0.6% 로 주차가 실측된 것의 처방

### 텀블러 자산 (2026-08-06 변경: 원기둥 근사가 기본)
    TUMBLER_URDF 기본 = assets/tumbler/tumbler_cyl_fric08.urdf
    = 원기둥 2개 근사 (캡 r50.2/h30mm, 몸체 R50/H239.4mm — STL 실측, 손잡이 제거,
      64각, 캡 관성은 균질 원기둥 공식) + free6 조인트 사슬 + 나사 저항 0.8Nm.
    나사 저항: URDF <dynamics friction> 을 screw_coupling 이 파싱해 "목표속도 0
    + 감쇠 1000 + effort=τ" 속도 드라이브로 구현 (PhysX 가 DOF friction 을
    무시하기 때문 — 토크 사다리로 문턱 실측 검증). 0.8 은 실물 "보통
    손조임"(0.5~1.5Nm) 중간, 사용자 지정값. 유효 μ1.5 에서 접선 16N =
    손가락당 ~2.7N 수직력 필요 — 서보 여력 안. 실측되면 그 값으로 갱신.
    생성기: assets/tumbler/make_cyl_tumbler.py (--r --h --R --H --spin-friction)
    보상·grasp_init 이 가정하는 원기둥과 물리 형상이 이것으로 처음 일치한다.
    주의: BPS obs 가 mesh 기반이라 실물 mesh 판 체크포인트(Main12 이하)와
    호환 안 됨. Main12 재현은 TUMBLER_URDF=tumbler_free6.urdf +
    HAND_FRICTION=4.0 OBJ_FRICTION=2.0.

### 해제 래치 판정 (2026-08-06 수정, 공용 screw_coupling)
    해제 래치는 누적각과 **원시 조인트각** 양쪽으로 건다 (둘 중 하나가
    engage 도달 시 해제). 누적각 적분은 리셋 정착 중의 캡 미세 회전·
    source_lower 클램프·한 스텝 지연이 겹쳐 원시각보다 ~3도 뒤처질 수 있고,
    allegro Al5 에서 캡이 조인트 한계(182도)에 물리적으로 닿아도 누적각이
    179.x 에 머물러 180도 래치가 영영 안 걸리는 것으로 실측됐다. cap_spin 은
    한계 182도 < 360도라 랩이 불가능하므로 원시각이 참값이다. dg5fs 는
    그립이 약해 증상이 없었지만 같은 수정의 수혜를 받는다 (판정 통일).
    진단 흔적: ENGAGE_DEG=179 "공차"는 이 오프셋에 대한 붕대였고 철회됨.

### free6 조인트 사슬 (2026-08-05)
    (실물 mesh 판 tumbler_free6.urdf 에서 도입, 원기둥 판도 동일 사슬 사용)
    캡을 6자유도 사슬(spin+lift+tx/ty/rx/ry)로 연결. 잠긴 동안 여분 4개는
    screw_coupling 이 강성 드라이브로 0 고정(측면 20N 에 0.04mm) → 기존
    2자유도 판과 같은 거동. 해제 순간 드라이브를 끄고 spin 한계도 풀어
    캡이 완전 자유(기울임·수평·계속 회전). 2자유도 판(tumbler.urdf)은
    해제 후 z 회전+상승만 가능했음 — GUI 로 확인된 비현실 거동.
    Main11 이하는 구판. 구 tools 재현은 URDF 에 여분 조인트가 없어 영향 없음.
    free6 해제 시 lift 하한도 -0.35 로 열려 캡이 진짜로 떨어진다 (구판의
    "해제 높이 바닥"은 free6 에서 제외). 낙하 종료: 캡이 홈 -30mm 아래로
    가면 에피소드 즉시 종료 (CAP_UNSCREW_DROP_FAIL_Z=0.03, env 기본 꺼짐).
    released 가 래치라 떨어진 캡을 바닥에서 재파지하면 성공·수입이 복구되는
    구멍(주차류 유인)을 막는 장치. 성공 시에는 종료하지 않음(주차 방지 유지).

### 접촉 마찰 (2026-08-06 변경)
    HAND_FRICTION=1.5, OBJ_FRICTION=1.5 → 유효 μ = 1.5 (PhysX 평균 결합)
    실리콘 손끝-플라스틱 캡의 문헌 중심치. 구판 하드코딩 4.0/2.0(유효 3.0)은
    "깨끗한 부드러운 실리콘" 낙관치 — Main12 이하는 그 값으로 학습됨.
    env 코드 기본은 4.0/2.0 유지 (구 재현 보존), 런처가 1.5 로 내림.
    rolling(0.01/0.05)·torsion 은 종전 그대로.

### 승격 기록 (2026-08-06, Main12 sr 99.5~100% 검증 후 기본값 반영)
    CLOSURE_AREA=1 + OPPOSE_W=0    Main11 에서 A/B (동일 시점 100% 수렴)
    ACTION_DELTA_BASE=target, SCALE=0.05, DOF 300/18/2.0   Main8~12
    INIT_POOL=poses/pool_m05.pt    Main12 (생성기 grasp_init/make_init_pool.py)
    → 이제 인자 없는 학습·PLAY 실행이 곧 Main12 구성이다. 학습과 GUI 재현이
      같은 세팅이 되도록 오버라이드 없이 쓰는 것이 원칙.
    예외: 접촉 마찰만 Main12(4.0/2.0) 이후 1.5/1.5 로 내려감 — 위 마찰 절 참조.
    Main12 원판 재현: HAND_FRICTION=4.0 OBJ_FRICTION=2.0 만 붙이면 된다.

## 폐기된 런 기록
    Main1  +0.3/z+10 초기값 (깨짐)          Main2  계단 접촉 게이트
    Main3  래치 없음 → 174도 주차 학습       Main4  현 기준 (ep500 warm start)
    Main10 넓이 closure 인접 3짝 판 — ep243 에서 6짝 판(Main11)으로 대체
