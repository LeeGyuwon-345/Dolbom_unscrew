# tools2_allegro — Allegro Hand (오른손) 캡 unscrew

> tools2(dg5fs) 스택의 allegro 이식판. 기준 구성 규율은 tools2/BASELINE.md 와
> 동일: **인자 없는 `./train_cap_unscrew_allegro.sh` = 기준**, 런처 기본값
> 변경 시 이 문서 동시 갱신.

손 모델: [simlabrobotics/allegro_hand_ros] 의 allegro_hand_description_right
— ManipTrans 내장 `assets/allegro_hand/allegro_hand_right.urdf` 가 관절
구조·한계 완전 동일함을 확인하고 내장본을 사용한다 (`dexhand=allegro`).
16관절 4지 (엄지 + 검지/중지/약지), 접촉 슬롯은 env 규약상 5칸이며
**5번째 = 약지 중복** — 보상이 `CAP_DUP_LAST_FINGER=1` 로 이중 계산을 막는다.

## tools2 와 다른 점

    dexhand              allegro (dg5fs 아님)
    손끝 팜 축            distal 링크 로컬 +x (실측; dg5fs 는 +y)
                          → CAP_PALM_LOCAL_AXIS=1,0,0
    손바닥 법선           base_link +x — dg5fs 손목 규약과 우연히 동일,
                          topdown 손목 샘플러 무수정 사용
    손끝 반경             0.012 m (URDF tip mesh 12mm)
    자기충돌 마스크        비움 (dg5fs 의 link_base/link_1_1/link_1_2 는 무의미.
                          allegro 유령 접촉은 아직 미조사 — 아래 TODO)
    초기자세              없음 → INIT_FRAC=0, topdown 샘플러로 시작
                          (grasp_init 재최적화 전까지)
    보상/물리             tools2 기준 그대로 (원기둥+0.8Nm, μ1.5, free6,
                          낙하종료, 해제+0.5초 유지 성공, 6짝→3짝 closure 자동)

## 이식 중 잡은 이슈 (Al1~Al9 이력)

    Al1~Al3  손-텀블러 충돌이 통째로 꺼져 있었음 — allegro 자기충돌 필터(1)와
             텀블러 필터(1)의 AND 겹침. 텀블러를 2 로 옮겨 해결 (env 수정)
    Al5      캡이 한계 182도까지 돌아도 누적각이 179.x — 적분 오프셋으로 180도
             래치 불가. 원시 조인트각 래치(공용 screw_coupling)로 근본 수정
    Al6      ENGAGE 179 공차(임시 붕대) + 성공판정의 각도 이중조건 버그 발견
             → 성공 = "해제 후 0.5초 유지 타이머"만으로 판정하게 수정
    Al9      관절 토크 0.7Nm (Allegro V4 스펙; 2.0 은 dg5fs 값의 오상속) —
             실기 사양으로 ep300 에 100% 수렴, 최종 검증판
    자기충돌  켬 (필터 0). rest·파지 유령 접촉 0 실측, 마스크 불필요

## TODO (순서 제안)

1. **grasp_init allegro 판**: optimize_pose 를 allegro FK(팁 link_X.0_tip,
   팜 축 +x, 16관절+손목6=22차원)로 이식해 파지 자세 최적화 → 노이즈 풀 생성
   → INIT_FRAC=1.0 전환. dg5fs 경험상 이게 학습 성패를 갈랐다.
2. 자기충돌 유령 접촉 조사 (dg5fs 는 엄지-손바닥 13554N 유령이 있었다 —
   allegro 도 rest 자세 접촉력 덤프로 확인할 것).
3. closure AREA_REF 재보정 (3짝 + allegro 손가락 간격 기준. 최적화 자세가
   나오면 그 자세의 넓이로 스케일).
4. 굽힘 방향·관절 한계는 URDF 가 실기 스펙이라 dg5fs 같은 placeholder 문제
   없음 (한계 비대칭, 실측치) — 별도 검증 불필요할 것으로 보이나 GUI 로 확인.

## 실행

    ./train_cap_unscrew_allegro.sh                    # TAG 기본 Al1
    TAG=Al1 MAX_ITER=10000 ./train_cap_unscrew_allegro.sh
    PLAY=1 GUI=1 CKPT=<.pth> ./train_cap_unscrew_allegro.sh

실험명 CapUnscrewAllegro_*, 로그 cap_unscrew_allegro_*.log — dg5fs 런과
섞이지 않는다. 스모크(8env 2epoch) 2026-08-06 통과.
