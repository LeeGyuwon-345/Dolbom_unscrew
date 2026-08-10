#!/usr/bin/env bash
set -euo pipefail

cd /home/leegyuwon/Documents/ManipTrans

export PATH="/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin:${PATH}"
export HYDRA_FULL_ERROR=1
export CAP_GRASP_MODE=1
export CAP_GRASP_TOOLS_DIR=/home/leegyuwon/Documents/task1/tools
export CAP_GRASP_URDF=/home/leegyuwon/Documents/task1/assets/cap_only/cap_only.urdf
export CAP_GRASP_WRIST_SAMPLER=mcp_reachable
export CAP_REQUIRE_CAP_CENTER_IN_ZERO_TRIANGLE=1
export CAP_PALM_DOWN_MAX_DEG=20
export CAP_WRIST_Z_MIN=0.040
export CAP_WRIST_Z_MAX=0.095
export CAP_THUMB_MCP_OUTSIDE_MARGIN=none
export CAP_REACH_CONTACT_MARGIN=0.020
export CAP_REACH_MIN_FINGERS=2
export CAP_REACH_REQUIRE_THUMB=1
export CAP_THUMB_REACH_MARGIN=0.025
export CAP_GRASP_CAP_LOCK_MODE=teleport
export CAP_GRASP_RESAMPLE_WRIST_ON_RESET=0

/home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python main/rl/train.py \
  task=ObjDexUnscrew \
  side=RH \
  dexhand=dg5fs \
  experiment=CapGraspDG5FS_TriangleXYReach_Z040_GUI \
  num_envs=4 \
  max_iterations=10 \
  test=true \
  headless=false \
  dataIndices='[7]' \
  checkpoint=/home/leegyuwon/Documents/ManipTrans/runs/CapGraspDG5FS_TriangleXYReach_Z040_10ep__07-29-01-43-47/nn/last_CapGraspDG5FS_TriangleXYReach_Z040_10ep_ep_10_rew__-0.19_.pth
