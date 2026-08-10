# grasp_init 패키지

dg5fs 5지 로봇손이 반지름 r, 높이 h 인 원기둥(텀블러 캡)을 잡는 26차원
자세(손가락 관절 20 + 손목 위치 3 + 자세 쿼터니언)를 최적화로 찾는 도구.

## 구성

    grasp_init/
      cap_shape.py            캡 = r x h 원기둥 (기본 r=50mm, h=17mm)
      optimize_pose.py        26차원 최적화 (배치 Adam, 미분가능 FK)
      maniptrans_dof_order.py URDF 에서 뽑은 관절 순서·가동범위
      reference_pose.py       자세 저장/로드 형식
      view_pose.py            matplotlib 3D 렌더 (자세+원기둥)
      view_pose_gym.py        Isaac Gym 뷰어 (선택 -- isaacgym 필요)
      view_cap.py             원기둥 vs 실제 캡 메시 대조
      poses/                  최적화 결과 (.json)
    assets/
      dg5fs_hand/             오른손 URDF + 링크 STL
      tumbler/                텀블러 URDF + 메시 (뷰어 대조용)

경로는 전부 패키지 상대라 아무 데나 풀어도 된다.

## 의존성

    필수:  torch, pytorch_kinematics, trimesh, scipy, numpy
    선택:  matplotlib (view_pose/view_cap), open3d (메시 데시메이션),
           isaacgym (view_pose_gym)

## 실행

    cd grasp_init
    python optimize_pose.py -r 0.050 -H 0.017 -n 96 --iters 1500 \
        -o poses/my_grasp.json
    python view_pose.py poses/my_grasp.json          # 렌더 확인

## 목적함수 (optimize_pose.objective)

    pad     손끝 패드 면(손바닥 축 +y 로 8.1mm)과 원기둥 옆면 거리 -> 0
    facing  손끝 법선이 캡 축을 향하게 (1-cos)^2, 학습 판정과 같은 부호 규약
    under   손의 어느 점도 원기둥 밑면 아래로 못 감 (최대깊이 벌점)
    area    엄지 포함 삼각형 3개 넓이 합 최대화 (대향+펼침)
    pen     tip 제외 링크의 원기둥 침투 금지 (최대깊이)
    selfc   손가락 캡슐(선분+반경) 간 4mm 이상 (자기충돌)

주의점과 실패 사례는 optimize_pose.py 첫머리 docstring 에 정리돼 있다
(평균 집계의 함정, 성긴 샘플, 힌지 경계 안착 등).
