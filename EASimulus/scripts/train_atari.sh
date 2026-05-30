export WANDB__SERVICE_WAIT=120
export CUDA_LAUNCH_BLOCKING=0 

envs=(Breakout Alien Amidar Assault Asterix BankHeist ) 
seeds=(0)
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python src/main.py tokenizer.image.with_lpips=True benchmark=atari env.train.id=${env}NoFrameskip-v4 wandb.name=${env}-seed${seed} \
        common.seed=${seed} world_model.event_pred=True world_model.ges=True wandb.mode=online \
        wandb.project=PROJECT_NAME_EASimulus;
    done
done

#By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off or `wandb.mode=offline` for offline logging.
