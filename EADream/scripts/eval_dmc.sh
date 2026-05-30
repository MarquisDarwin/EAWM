#!/usr/bin/env bash
cd ..

LOGROOT="log"
CONFIG=configsc.yaml
CONFIGS=dmc_vision
EPISODES=10
WEIGHTS_DIR= path/to/checkpoint/dir/EADream
RESULT_FILE=dmcresult.txt
DEVICE=cuda:0
TASKS=(
  dmc_acrobot_swingup
  dmc_cartpole_swingup_sparse
  dmc_cheetah_run
  dmc_finger_turn_hard
  dmc_hopper_hop
  dmc_quadruped_run
  dmc_quadruped_walk
  dmc_reacher_hard
)
mkdir -p "${WEIGHTS_DIR}" "$(dirname "${RESULT_FILE}")"


for task in "${TASKS[@]}"; do
  weights="${WEIGHTS_DIR}/${task}.pt"
  if [[ ! -f "${weights}" ]]; then
    echo "skip ${task}: missing weights ${weights}"
    continue
  fi
  echo "start ${task} weights=${weights}"
  python3 eval.py \
    --configdir "${CONFIG}" \
    --configs "${CONFIGS}" \
    --task "${task}" \
    --weights "${weights}" \
    --episodes "${EPISODES}" \
    --device "${DEVICE}" \
    --result-file "${RESULT_FILE}"
done
