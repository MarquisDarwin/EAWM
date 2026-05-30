#!/usr/bin/env bash
GAMES=(Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown)
CHECKPOINT_DIR=/path/to/checkpoint/dir/EASimulus/Atari
LOG_DIR=outputs/atari_eval_logs
NUM_EPISODES=100
NUM_ENVS=20
SEED=0
WANDB_MODE=disabled
mkdir -p ${LOG_DIR}
for game in "${GAMES[@]}"; do
    weights_path="${CHECKPOINT_DIR}/${game}.pt"
    log_file="${LOG_DIR}/${game}.log"

    if [ ! -f "${weights_path}" ]; then
        echo "===== ${game}: missing checkpoint ${weights_path} ====="
        continue
    fi

    echo "===== Evaluating ${game} (${weights_path}) ====="
    python scripts/eval.py \
        --benchmark atari \
        --weights-path "${weights_path}" \
        --num-episodes "${NUM_EPISODES}" \
        --num-envs "${NUM_ENVS}" \
        --seed "${SEED}" \
        --wandb-mode "${WANDB_MODE}" 2>&1 | tee "${log_file}"
done

