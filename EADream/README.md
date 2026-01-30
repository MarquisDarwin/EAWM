# EADream
The base world model of EADream comes from [Pytorch implementation](https://github.com/NM512/dreamerv3-torch/issues/65) of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). 

## Environment Setup
- Create a conda environment:
```
conda create -n EADream python=3.9 -y
```

Get dependencies with python 3.9:
```
pip install 'pip<24.1'
pip install -r requirements.txt
```


## Launch a Training Run

#### Atari 100K

To Import ROMs, use[scripts/import_atari_rom.sh](scripts/import_atari_rom.sh):
''`
cd scripts
chmod +x import_atari_rom.sh
./import_atari_rom.sh
```

Train the agent using [scripts/train_atari.sh](scripts/train_atari.sh) on the Atari 100K benchmark:
```
cd scripts
chmod +x train_atari.sh
./train_atari.sh
```

#### DMC Vision 500K
Run training on DMC Vision:
```
python3 dreamer.py --configdir configsc.yaml --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

#### DMC-GB2 500K
The [DMC-GB2 benchmark](https://github.com/aalmuzairee/dmcgb2) has dependencies on external datasets. You need to you need to download the Places365 Dataset and DAVIS Dataset:
```
mkdir -p envs/data;
cd envs/data;
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip;
unzip DAVIS-2017-trainval-480p.zip;
rm DAVIS-2017-trainval-480p.zip;
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar;
tar -xvf places365standard_easyformat.tar; 
rm places365standard_easyformat.tar;
```
The folder ```envs/data``` is structured as
```
envs/data
└─── places365_standard
└─── video_easy
└─── video_hard
└─── color_easy.pt
└─── color_hard.pt
```

Run training on DMC-GB2 Benchmark, use [scripts/train_dmcgb2.sh](scripts/train_dmcgb2.sh):
``` 
cd scripts
chmod +x train_dmcgb2.sh
./train_dmcgb2.sh
```
## Logging and Monitoring

Monitor results with tensorboard:
```
tensorboard --logdir ./logdir
```


