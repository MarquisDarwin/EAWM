#!/usr/bin/env bash
LOG_DIR=outputs/craft_eval_logs
NUM_EPISODES=100
NUM_ENVS=20
SEED=0
WANDB_MODE=disabled
weights_path=/path/to/checkpoint/dir/EASimulus/craftax.pt
log_file="${LOG_DIR}/craftax.log"
mkdir -p ${LOG_DIR}
if [ ! -f "${weights_path}" ]; then
    echo "===== missing checkpoint ${weights_path} ====="
    continue
fi

echo "===== Evaluating craftax (${weights_path}) ====="
python scripts/eval.py \
    --benchmark craftax \
    --weights-path "${weights_path}" \
    --num-episodes "${NUM_EPISODES}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --wandb-mode "${WANDB_MODE}" 2>&1 | tee "${log_file}"


