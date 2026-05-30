seeds=(2)
export OPENBLAS_NUM_THREADS=1 
export HYDRA_FULL_ERROR=1
for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/main.py benchmark=craftax wandb.name=craftax-seed${seed} common.seed=${seed} actor_critic.intrinsic_reward_weight=1\
    world_model.motion_pred=True world_model.amas=True  world_model.reward_end_use_all_embedding=True world_model.judge_tendency=True

done