# EASimulus 
The base world model of EASimulus comes from [the official implementation](https://github.com/leor-c/Simulus) of Simulus.

## Environment Setup
We recommend following the steps to set up the environment. 

- Create a conda environment:
``` 
conda create -n EASimulus python=3.10 -y
```

  - Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with torch==2.5.1, but should work with other recent version.
  - Install [other dependencies](requirements.txt):
  ``` 
  pip install -r requirements.txt
  ```

- Download pretrained VGG weights for LPIPS:
```
python get_lpips.py
```


##  Launch a Training Run

We provide a [training script](scripts/train.sh) for the Atari 100K benchmark. To run the script, use:
```bash
chmod +x scripts/train.sh
./scripts/train.sh
```


To change an environment within a benchmark, set `env.train.id` by modifying the appropriate configuration file located in `config/env` or through the command line:
```bash
python src/main.py benchmark=atari env.train.id=BreakoutNoFrameskip-v4
```
To run the Craftax benchmark, use `benchmark=craftax` for Craftax.
To turn off the event prediction and GES, use:
 ```world_model.event_pred=False``` and ```world_model.ges=False```

## Logging and Monitoring

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off 
or `wandb.mode=offline` for offline logging.







