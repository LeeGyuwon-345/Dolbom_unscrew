# rh_init — (r,h) → allegro 초기 파지자세

원기둥(뚜껑) 반지름 r·높이 h 를 넣으면 allegro 손의 초기 파지자세를 내는 모델.
`grasp_init/optimize_pose_allegro.py`(per-instance 최적화)를 **amortize(학습으로 대체)** 한다.

## 핵심 설계: 방위각 정규화
원기둥은 축대칭 → 파지 방위각 자유도가 무한. (r,h)마다 방위각이 다르면 회귀
타깃이 다봉이라 학습이 망가진다. 그래서 **손목을 원통 +x 축(azimuth=0)으로 고정**한
정규 자세만 학습한다. 손가락 `dof_pos` 는 방위각 무관. 배포 시 원하는 θ 로
`model.rotate_pose` 회전해서 놓는다.

자세 = `dof_pos(16) + wrist_rel_pos(3) + wrist_quat(4)` — `grasp_init/reference_pose.GraspInitPose`
와 동일 규약(Isaac Gym DOF 순서: index→thumb→middle→ring). 학습/뷰어에 그대로 소비.

## 파일
| 파일 | 역할 |
|---|---|
| `model.py` | `GraspMLP`(2→128→128→24, 손목 azimuth=0 강제) + 6D회전 유틸 + `rotate_pose` |
| `objective_batched.py` | per-sample (r,h) 미분가능 objective (pad/facing/under/area/pen/selfc). `optimize_pose_allegro` FK·헬퍼 재사용 |
| `train_amortized.py` | **[권장 B]** MLP 가 낸 자세에 objective 를 걸어 최소화. 옵티마이저 데이터 불필요 |
| `lookup.py` | **[베이스라인]** (r,h) 격자에 옵티마이저 자세를 풀어 저장 + 쌍선형 보간 |

## r,h 범위 (★ 나중에 입력)
`train_amortized.py` 상단 `R_RANGE`, `H_RANGE` (현재 플레이스홀더 30~70mm / 10~30mm).
`lookup.py` 는 이 값을 공유한다.

## 사용
```bash
# [B] amortized 학습 (범위 먼저 채우고)
python train_amortized.py --iters 20000 --batch 256
python train_amortized.py --export 0.05 0.017 -o out.json   # 자세 생성

# [베이스라인] 격자 룩업 (느림: 격자점마다 옵티마이저 1회)
python lookup.py build --nr 9 --nh 6
python lookup.py show 0.05 0.017
```

## 진행 순서 (추천)
1. **lookup** 으로 베이스라인 (2D 매끄러움이라 이것만으로 충분할 수 있음).
2. 부족하면 **amortized(B)** — objective 재사용, 데이터 불필요, 품질↑.
3. 최고 품질: lookup 데이터로 사전학습 → objective 미세조정(하이브리드).

## 검증 상태
- amortized 학습이 수렴함 확인 (facing 항이 iter 2000에 ~0.09 = cos 0.77 로, per-instance
  옵티마이저(0.044)에 접근). 전량(20000 iter) 학습 시 옵티마이저 품질 도달 예상.
- pad/pen/selfc/under = 0 유지 (관통·자기충돌 없이 벽 밀착).

## 주의
- 생성 자세는 **정규 방위각(손목 +x)**. 실제 배치 방위각은 `rotate_pose(θ)` 후처리.
- DOF 순서는 Isaac Gym 순서(index→thumb→middle→ring) — `optimize_pose_allegro` 와 동일.
