# tools2 — 두 번째 강화학습

> **기준 구성은 BASELINE.md 다.** 인자 없는 `./train_cap_unscrew2.sh` = 기준.
> 코드를 고칠 때는 BASELINE.md 의 규칙(런처 기본값이 기준, 문서 동시 갱신)을
> 따른다 — 수정 때마다 방향이 흐르는 것을 막는 장치다.

환경·씬은 기존(tools)과 동일하고, 세 가지가 다르다.

## 1. 보상/성공 조건 — 이 폴더에서 재작성

환경(dexhandmanip_sh)은 `CAP_GRASP_TOOLS_DIR` 가 가리키는 폴더에서 보상 모듈을
import 한다. 이 폴더를 가리키면 `cap_unscrew_curriculum_terms.py` 가 tools 판
대신 로드되고, 여기 없는 공용 모듈(screw_coupling, wrist_init, cap_grasp_rl_terms,
cap_unscrew_rl_terms)은 tools 로 fallback 한다 — 환경에 그 fallback 을 넣어
두었으므로 공용 모듈을 복사할 필요가 없다.

지금 들어 있는 것은 tools 판의 사본이다. 성공 조건과 보상 구성은 여기서 다시
짠다. 어디를 고치면 되는지:

    CapUnscrewCurriculumConfig     가중치·임계값 (환경변수로 오버라이드됨)
    compute 함수의 objective 블록  보상 항 조합
    success = ...                  성공 판정

## 2. 파지 초기자세 — 최적화한 26차원을 리셋에 입력

`poses/grasp_26d.json` = grasp_init 최적화 결과 (pad_area4). 다섯 패드가 벽에
접촉하고(4개는 0.5mm 이내), 방위각 224도 분산, 캡슐 자기충돌·침투·밑면 침범
모두 0 인 자세다. 26차원 = 손가락 관절 20 + 손목 상대위치 3 + 자세(쿼터니언
저장) — 손목은 캡 기준 상대라 캡이 어디 있든 그대로 쓰인다.

리셋 배선은 grasp_init/reference_pose.py 가 담당하고 (이미 환경에 연결돼
있음), 런처가 환경변수만 채운다:

    CAP_GRASP_INIT_POSE    자세 파일. 비우면 기능 꺼짐
    CAP_GRASP_INIT_FRAC    이 자세로 시작할 환경 비율 (기본 0.5)
    CAP_GRASP_INIT_NOISE   관절각 가우시안 노이즈 rad (기본 0.05)

기본 0.5 인 이유: 전부 같은 자세에서 출발시키면 그 자세 전용 정책이 된다.
나머지 절반은 기존 리셋(topdown 손목 샘플링)을 그대로 탄다.

## 3. 실행

    ./train_cap_unscrew2.sh                  # TAG 기본 Rl2
    MAX_ITER=1000 ./train_cap_unscrew2.sh
    PLAY=1 CKPT=<.pth> ./train_cap_unscrew2.sh
    INIT_FRAC=0 ./train_cap_unscrew2.sh      # 초기자세 끄기

로그는 task1/logs/cap_unscrew2_<TAG>_*.log, 실험명은 CapUnscrew2DG5FS_* 라
기존 런과 섞이지 않는다.

## 주의

- 보상을 고치면 tools2 쪽만 고칠 것. tools 판은 기존 체크포인트(Geo4, Turn)의
  재현용으로 그대로 둔다.
- 관측(cap_phase 등)은 환경 소유라 여기서 못 바꾼다. 관측을 바꾸면 기존
  체크포인트와 호환이 깨진다는 것도 같이 기억할 것 (HIDE_ANGLE 은 차원을
  유지하므로 예외).
