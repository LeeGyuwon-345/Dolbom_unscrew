#!/bin/bash
# 학습된 MLP로 (r,h) 파지자세 생성 후 바로 GUI 뷰어 실행.
#   사용: bash show.sh <r_mm> <h_mm>
#   예:   bash show.sh 30 25        # 반지름 30mm, 높이 25mm
# 학습 범위: r=15~55mm, h=20~30mm (벗어나면 MLP가 외삽이라 품질 저하).
set -u
cd /home/leegyuwon/Documents/task1/rh_init
export PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin:${PATH:-}
export LD_LIBRARY_PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib:${LD_LIBRARY_PATH:-}
PY=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python

R_MM=${1:?"r(mm) 필요. 예: bash show.sh 30 25"}
H_MM=${2:?"h(mm) 필요. 예: bash show.sh 30 25"}
R_M=$($PY -c "print($R_MM/1000.0)")
H_M=$($PY -c "print($H_MM/1000.0)")
OUT=/tmp/mlp_r${R_MM}_h${H_MM}.json

echo "[show] MLP로 r=${R_MM}mm h=${H_MM}mm 자세 생성..."
$PY train_amortized.py --export "$R_M" "$H_M" -o "$OUT" || exit 1

echo "[show] GUI 뷰어 실행 (--collide). SPACE 수치, ESC 종료."
$PY /home/leegyuwon/Documents/task1/tools2_allegro/view_pose_allegro.py "$OUT" --collide
