cd ..
savedir="logdir" 
seeds=(0)
config="configsc.yaml"
mkdir ${savedir}
envs=(cartpole_swingup) #walker_stand walker_walk  finger_spin cheetah_run  cup_catch
cp ${config} ./${savedir}/${config}
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python3 dreamergb.py --configdir ${config} --configs dmc_vision --task dmcgb_${env} --seed ${seed} --logdir ./${savedir}/dmc_${env}-${seed};
    done
done
