#!/bin/bash
# B03 체크포인트/조건 지정 재생:  ./play_b03_at.sh <ep> [저항Nm] [마찰μ]
# 예) ./play_b03_at.sh 1800 1.7 5.0     (그 체크포인트가 학습하던 조건)
#     ./play_b03_at.sh 20000            (기본 = 저항4.0 μ2.0 최종 조건)
EP=${1:?ep 필요}; R=${2:-4.0}; MU=${3:-2.0}
RUN=$(ls -dt /home/leegyuwon/Documents/ManipTrans/runs/CapUnscrewArm_B03_2026* | head -1)
CK=$(ls $RUN/nn/last_*ep_${EP}_*.pth 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "ep${EP} 체크포인트 없음"; exit 1; }
SC=$(mktemp -d)
sed -e "s|^CKPT=.*|CKPT=$CK|" \
    -e "s/CAP_RESIST_START=0.2/CAP_RESIST_START=$R/" -e "s/CAP_RESIST_END=[0-9.]*/CAP_RESIST_END=$R/" \
    -e "s/CAP_FRICTION_CURR_START=5.0/CAP_FRICTION_CURR_START=$MU/" -e "s/CAP_FRICTION_CURR_END=[0-9.]*/CAP_FRICTION_CURR_END=$MU/" \
    -e "s/HAND_FRICTION=5.0/HAND_FRICTION=$MU/" -e "s/OBJ_FRICTION=5.0/OBJ_FRICTION=$MU/" \
    /home/leegyuwon/Documents/task1/tools2_allegro/play_b03.sh > $SC/p.sh
chmod +x $SC/p.sh
echo "[play] ep${EP}  저항 ${R}N·m  마찰 ${MU}"
exec $SC/p.sh
