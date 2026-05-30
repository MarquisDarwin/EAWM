#!/usr/bin/env bash
cd ..

CONFIG=configsc.yaml
EPISODES=100
DEVICE=cuda:0
RESULT_FILE=log/result.txt
WEIGHTS_DIR= path/to/checkpoint/dir/EADream
GAMES=(
  asterix
  bank_heist
  battle_zone
  boxing
  breakout
  chopper_command
  crazy_climber
  demon_attack
  freeway
  frostbite
  gopher
  hero
  james_bond
  kangaroo
  krull
  kung_fu_master
  ms_pacman
  pong
  private_eye
  qbert
  road_runner
  seaquest
  up_n_down
)
for game in "${GAMES[@]}"; do
  weights="${WEIGHTS_DIR}/atari_${game}.pt"
  echo "start ${game} weights=${weights}"
  python3 eval.py \
    --configdir "${CONFIG}" \
    --game "${game}" \
    --weights "${weights}" \
    --episodes "${EPISODES}" \
    --device "${DEVICE}" \
    --result-file "${RESULT_FILE}"
done

