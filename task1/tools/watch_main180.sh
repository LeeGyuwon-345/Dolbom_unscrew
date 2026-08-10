#!/usr/bin/env bash
# Keep the Main180 run alive to 30000 epochs.
#
# The run is ~20 hours and has already died twice for reasons that had nothing
# to do with the policy: a CUDA illegal-memory-access when leftover PLAY
# processes shared the 8 GB card, and a NameError from a diagnostic edit. Both
# were recoverable from the last checkpoint, so this restarts from it rather
# than losing the run to an unattended crash overnight.
#
# Deliberately not a blind loop: it gives up after MAX_RESTARTS so a fault that
# reproduces every time stops instead of burning the night on restart cycles,
# and it never restarts a run that reached the target.
set -uo pipefail

TOOLS=/home/leegyuwon/Documents/task1/tools
RUNS=/home/leegyuwon/Documents/ManipTrans/runs
LOGS=/home/leegyuwon/Documents/task1/logs
TAG="${TAG:-Geo1}"
TARGET="${TARGET:-30000}"
MAX_RESTARTS="${MAX_RESTARTS:-8}"
WLOG="${LOGS}/watchdog_${TAG}.log"

say() { echo "[$(date +%H:%M:%S)] $*" >> "$WLOG"; }

trainer_pid() {
  for p in $(pgrep -f "main/rl/train.py" 2>/dev/null); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in
      /home/leegyuwon/*python*experiment=CapUnscrewCurrDG5FS_${TAG}_*) echo "$p"; return;;
    esac
  done
}

latest_log() { ls -t "${LOGS}/cap_unscrew_curr_${TAG}_"*.log 2>/dev/null | head -1; }
latest_ckpt() { ls -t "${RUNS}/CapUnscrewCurrDG5FS_${TAG}_"*/nn/*.pth 2>/dev/null | head -1; }

reached_target() {
  local l; l=$(latest_log); [[ -n "$l" ]] || return 1
  grep -q "epoch: ${TARGET}/${TARGET}" "$l"
}

# Epoch reached so far, across every restart of this tag.
epochs_done() {
  local best=0 n
  for l in "${LOGS}/cap_unscrew_curr_${TAG}_"*.log; do
    [[ -f "$l" ]] || continue
    n=$(grep -oE "epoch: [0-9]+/" "$l" | tail -1 | tr -cd '0-9')
    [[ -n "$n" && "$n" -gt "$best" ]] && best=$n
  done
  echo "$best"
}

restarts=0
say "watchdog 시작 (목표 ${TARGET}, 최대 재시작 ${MAX_RESTARTS})"

while :; do
  sleep 60

  if reached_target; then
    say "목표 도달 — watchdog 종료"
    exit 0
  fi

  [[ -n "$(trainer_pid)" ]] && continue

  # Gone but not finished.
  if (( restarts >= MAX_RESTARTS )); then
    say "재시작 ${MAX_RESTARTS}회 소진 — 포기. 로그 확인 필요"
    exit 1
  fi

  done_ep=$(epochs_done)
  ck=$(latest_ckpt)
  remain=$(( TARGET - done_ep ))
  (( remain < 1 )) && { say "남은 epoch 없음 — 종료"; exit 0; }

  restarts=$((restarts + 1))
  say "프로세스 없음 (epoch ${done_ep}). 재시작 ${restarts}/${MAX_RESTARTS}, 남은 ${remain}"
  [[ -n "$ck" ]] && say "  체크포인트: $(basename "$ck")" || say "  체크포인트 없음 — 처음부터"

  # A crash can leave the card allocated; clear anything of ours still on it.
  for p in $(pgrep -f "main/rl/train.py" 2>/dev/null); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in /home/leegyuwon/*python*) kill -9 "$p" 2>/dev/null;; esac
  done
  sleep 10

  cd "$TOOLS" || exit 1
  if [[ -n "$ck" ]]; then
    RESAMPLE_WRIST=1 MAX_ITER="$remain" NUM_ENVS=512 DIAG_EVERY=200 TAG="$TAG" CKPT="$ck" \
      PALM_FACING_COS="${PALM_FACING_COS:-0.8}" \
      setsid nohup ./train_cap_unscrew_curriculum.sh > /dev/null 2>&1 < /dev/null &
  else
    RESAMPLE_WRIST=1 MAX_ITER="$remain" NUM_ENVS=512 DIAG_EVERY=200 TAG="$TAG" \
      PALM_FACING_COS="${PALM_FACING_COS:-0.8}" \
      setsid nohup ./train_cap_unscrew_curriculum.sh > /dev/null 2>&1 < /dev/null &
  fi
  sleep 120
  [[ -n "$(trainer_pid)" ]] && say "  재시작 성공" || say "  재시작 실패 — 다음 주기에 재시도"
done
