#!/bin/bash
# MLP로 (r,h) 파지자세 + RB5 IK 생성 후, RB5+Allegro+텀블러 통합 뷰어 실행.
#   사용: bash show_full.sh <r_mm> <h_mm> [cap_z_mm=225] [--force]
#   예:   bash show_full.sh 50 30            # 캡 배치높이 225mm(기본)
#         bash show_full.sh 50 30 180        # 캡을 RB5 base 기준 z=180mm 에 배치
# cap_z_mm = 캡 월드 배치높이(RB5 base 기준 z). 범위 [140,230]mm. 손파지엔 무관,
#            RB5 6관절만 이 높이로 재계산. 텀블러 x,y 는 (0.6,0) 고정.
# 주의: RB5 IK(방위각 탐색)로 자세 생성에 ~45초. 같은 (r,h,z) 는 캐시 재사용.
set -u
cd /home/leegyuwon/Documents/task1/rh_init
export PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin:${PATH:-}
export LD_LIBRARY_PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib:${LD_LIBRARY_PATH:-}
PY=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python

R_MM=${1:?"r(mm) 필요. 예: bash show_full.sh 50 30"}
H_MM=${2:?"h(mm) 필요. 예: bash show_full.sh 50 30"}
CAP_Z_MM=${3:-225}
FORCE=${4:-}
$PY -c "z=$CAP_Z_MM; assert 140<=z<=230, 'cap_z_mm 범위는 [140,230]'" || exit 1
R_M=$($PY -c "print($R_MM/1000.0)")
H_M=$($PY -c "print($H_MM/1000.0)")
CAP_Z_M=$($PY -c "print($CAP_Z_MM/1000.0)")
OUT=/tmp/mlp_full_r${R_MM}_h${H_MM}_z${CAP_Z_MM}.json

if [[ -f "$OUT" && "$FORCE" != "--force" ]]; then
  echo "[show_full] 캐시 재사용: $OUT  (재생성은 끝에 --force)"
else
  echo "[show_full] MLP 파지 + RB5 IK 생성중 (캡배치 z=${CAP_Z_MM}mm)... (~45초)"
  $PY train_amortized.py --export "$R_M" "$H_M" --cap 0.6 0 "$CAP_Z_M" -o "$OUT" || exit 1
fi

echo "[show_full] 통합 뷰어 실행. 창이 뜨기까지 5~10초. SPACE 관절값, ESC 종료."
$PY view_full.py "$OUT"
