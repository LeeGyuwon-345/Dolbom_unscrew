#!/bin/bash
# 직접 최적화(MLP 아님)로 (r,h,위에서%) 파지 생성 + RB5 IK + 통합 뷰어.
#   사용: bash show_opt.sh <r_mm> <h_mm> [위에서%] [fingers]
#   예:   bash show_opt.sh 20 30              # r20 h30, 위에서 60%(기본)
#         bash show_opt.sh 50 30 40           # 위에서 40% 로 덮어쓰기
#         bash show_opt.sh 30 25 60 thumb,index,middle
# 위에서% 기본 60. fingers 미지정 시 r로 자동(엄지검지 항상, 중지 r>=25, 약지 r>=40).
# 같은 (r,h,%)는 캐시 재사용. 재생성은 맨 끝 --force.
set -u
cd /home/leegyuwon/Documents/task1/rh_init
export PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin:${PATH:-}
export LD_LIBRARY_PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib:${LD_LIBRARY_PATH:-}
PY=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python

R_MM=${1:?"r(mm) 필요. 예: bash show_opt.sh 20 30 60"}
H_MM=${2:?"h(mm) 필요. 예: bash show_opt.sh 20 30 60"}
TOP_PCT=${3:-60}                       # 위에서 몇 % (기본 60 고정)
FINGERS=${4:-auto}
FORCE=${5:-}
R_M=$($PY -c "print($R_MM/1000.0)")
H_M=$($PY -c "print($H_MM/1000.0)")
ZFRAC=$($PY -c "print(1-$TOP_PCT/100.0)")   # 위에서 TOP% = 바닥기준 1-TOP%
# 캡 월드 배치높이(RB5 base 기준 z). 환경변수 CAP_Z_MM (기본 225), 범위 [140,230]mm.
# 손파지엔 무관, RB5 6관절만 이 높이로 재계산.
CAP_Z_MM=${CAP_Z_MM:-225}
$PY -c "z=$CAP_Z_MM; assert 140<=z<=230, 'CAP_Z_MM 범위는 [140,230]'" || exit 1
CAP_Z_M=$($PY -c "print($CAP_Z_MM/1000.0)")

if [ "$FINGERS" = "auto" ]; then
  FINGERS=$($PY -c "r=$R_MM; f=['thumb','index']+(['middle'] if r>=25 else [])+(['ring'] if r>=40 else []); print(','.join(f))")
fi
OUT=/tmp/opt_r${R_MM}_h${H_MM}_top${TOP_PCT}_z${CAP_Z_MM}.json

if [[ -f "$OUT" && "$FORCE" != "--force" ]]; then
  echo "[show_opt] 캐시 재사용: $OUT (재생성은 끝에 --force)"
else
  echo "[show_opt] 최적화 r=${R_MM} h=${H_MM} 위에서${TOP_PCT}% 손가락[$FINGERS] 캡배치z=${CAP_Z_MM}mm ...(~1분)"
  $PY /home/leegyuwon/Documents/task1/grasp_init/optimize_pose_allegro.py \
    -r "$R_M" -H "$H_M" --z-frac "$ZFRAC" --fingers "$FINGERS" \
    --restarts 24 --iters 1000 -o "$OUT" || exit 1
  echo "[show_opt] RB5 IK(방위각+엘보우업) 부착..."
  $PY add_rb5.py "$OUT" --cap 0.6 0 "$CAP_Z_M" || exit 1
fi

echo "[show_opt] 통합 뷰어. SPACE 관절값, ESC 종료."
$PY view_full.py "$OUT"
