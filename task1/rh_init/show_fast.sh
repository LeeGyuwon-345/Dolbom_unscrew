#!/bin/bash
# 두 amortized 네트워크로 (r,h,캡배치z) -> 손16+팔6=22관절 즉시추론 후 통합 뷰어.
# IK 없음(sub-ms). 손 MLP(grasp_mlp.pt) + RB5 net(rb5_mlp.pt) 필요.
#   사용: bash show_fast.sh <r_mm> <h_mm> [cap_z_mm=225]
#   예:   bash show_fast.sh 50 30
#         bash show_fast.sh 20 30 180
set -u
cd /home/leegyuwon/Documents/task1/rh_init
export PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin:${PATH:-}
export LD_LIBRARY_PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib:${LD_LIBRARY_PATH:-}
PY=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python

R_MM=${1:?"r(mm) 필요. 예: bash show_fast.sh 50 30"}
H_MM=${2:?"h(mm) 필요. 예: bash show_fast.sh 50 30"}
CAP_Z_MM=${3:-225}
$PY -c "z=$CAP_Z_MM; assert 140<=z<=230, 'cap_z_mm 범위는 [140,230]'" || exit 1
R_M=$($PY -c "print($R_MM/1000.0)"); H_M=$($PY -c "print($H_MM/1000.0)"); Z_M=$($PY -c "print($CAP_Z_MM/1000.0)")
OUT=/tmp/fast_r${R_MM}_h${H_MM}_z${CAP_Z_MM}.json

echo "[show_fast] 즉시추론 (r=${R_MM} h=${H_MM} 캡배치z=${CAP_Z_MM}mm, 텀블러x=0.4)..."
$PY train_rb5.py --export "$R_M" "$H_M" "$Z_M" -o "$OUT" || exit 1
echo "[show_fast] 통합 뷰어(실리콘 팁 siltip2). SPACE 관절값, ESC 종료."
ROBOT_URDF=rb5_allegro_vmount_siltip2.urdf PAD_FR=0.0105 $PY view_full.py "$OUT"
