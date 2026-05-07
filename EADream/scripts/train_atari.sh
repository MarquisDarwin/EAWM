cd ..
savedir="logdir"
config="configsc.yaml"
seed=0
mkdir ${savedir}
envs=(boxing) # ms_pacman alien amidar assault asterix bank_heist battle_zone breakout chopper_command crazy_climber demon_attack freeway frostbite gopher hero james_bond kangaroo krull kung_fu_master  pong private_eye qbert road_runner seaquest up_n_down
cp ${config} ./${savedir}/${config}
for env in "${envs[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 dreamer.py --configdir ${config} --configs atari100k --task atari_${env} --seed ${seed} --logdir ./${savedir}/atari_${env}-${seed}
done
