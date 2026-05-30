<div align="center">

<h2>From Observations to Events: Event-Aware World Models for Reinforcement Learning</h2>

Zhao-Han Peng, Shaohui Li, Zhi Li, Shulan Ruan, Yu Liu, You He

Tsinghua University, Zhejiang university

**ICLR 2026**

<a href='https://arxiv.org/abs/2601.19336'><img src='https://img.shields.io/badge/ArXiv-2510.10125-red'></a> 
<a href='https://huggingface.co/darwin05/EAWM'><img src='https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow?logo=huggingface&logoColor=yellow'></a>


</div>

## Overview

**EAWM** (Event-Aware World Models) is an event-aware world model framework for reinforcement learning. In addition to conventional observation prediction, EAWM explicitly models events so that the world model can learn sparser and more interpretable environment dynamics. The repository includes experiments on Atari, DeepMind Control Suite, DMC-GB2, and Craftax.

This repository provides two implementations:

- `EADream`: a DreamerV3-style implementation for Atari 100K, DMC Vision, and DMC-GB2.
- `EASimulus`: a Simulus-based implementation for Atari 100K and Craftax.

The visualization files in this repository show policy rollouts, outputs from the automated event generator, and open-loop future-observation predictions from the world model. Raw experiment results are available in `results/`.


![results](results/results.png)
## Environment Setup

Please choose one of the instantiations of EAWM from EADream and EASimulus, and follow the instructions in the [EADream/README.md](EADream/README.md) or [EASimulus/README.md](EASimulus/README.md) to configure the environment.

## Atari

### Visualization of Policy

<table>
    <tr>
        <td><center><img src="visualization\atari\eval\pong.gif" alt="pong" style="width:256px"> </center></td>
    <td ><center><img src="visualization\atari\eval\freeway.gif" alt="freeway" style="width:256px"> </center></td>
    <td><center> <img src="visualization\atari\eval\breakout.gif" alt="breakout" style="width:256px"></center></td>
        <td><center> <img src="visualization\atari\eval\qbert.gif" alt="james_bond" style="width:256px"> </center></td>
    </tr>
    
</table>


### Automated Event Generator

<table>
    <tr>
        <td><center><img src="visualization\atari\event\pong.gif" alt="pong" style="width:256px"> </center></td>
    <td ><center><img src="visualization\atari\event\freeway.gif" alt="freeway" style="width:256px"> </center></td>
    <td><center> <img src="visualization\atari\event\breakout.gif" alt="breakout" style="width:256px"></center></td>
        <td><center> <img src="visualization\atari\event\qbert.gif" alt="james_bond" style="width:256px"> </center></td>
    </tr>
</table>

### Imagination of Future Observations

<table>
    <tr>
        <td><center><img src="visualization\atari\observation\pong.gif" alt="pong" style="width:512px"> </center></td>
    <td ><center><img src="visualization\atari\observation\freeway.gif" alt="freeway" style="width:512px"> </center></td>
        </tr>
    <tr>
    <td><center> <img src="visualization\atari\observation\breakout.gif" alt="breakout" style="width:512px"></center></td>
        <td><center> <img src="visualization\atari\observation\qbert.gif" alt="james_bond" style="width:512px"> </center></td>
    </tr>
</table>

## DMC-GB2

### Visualization of Policy

<table>
    <tr>
        <td><center><img src="visualization\dmcgb2\eval\cartpole_swingup_EAWM.gif" alt="cartpole_swingup_EAWM.gif" style="width:256px"> </center></td>
    <td ><center><img src="visualization\dmcgb2\eval\cup_catch_EAWM.gif" alt="cup_catch_EAWM.gif" style="width:256px"> </center></td>
    <td><center> <img src="visualization\dmcgb2\eval\finger_spin_EAWM.gif" alt="finger_spin_EAWM.gif" style="width:256px"></center></td>
        <td><center> <img src="visualization\dmcgb2\eval\walker_stand_EAWM.gif" alt="walker_stand_EAWM.gif" style="width:256px"> </center></td>
    </tr>
</table>

### Automated Event Generator
<table>
    <tr>
        <td><center><img src="visualization\dmcgb2\event\cartpole_swingup.gif" alt="cartpole_swingup_EAWM.gif" style="width:256px"> </center></td>
    <td ><center><img src="visualization\dmcgb2\event\cup_catch.gif" alt="cup_catch_EAWM.gif" style="width:256px"> </center></td>
    <td><center> <img src="visualization\dmcgb2\event\finger_spin.gif" alt="finger_spin_EAWM.gif" style="width:256px"></center></td>
        <td><center> <img src="visualization\dmcgb2\event\walker_stand.gif" alt="walker_stand_EAWM.gif" style="width:256px"> </center></td>
    </tr>
</table>


### Open-loop prediction in train environments
<table>
    <tr>
        <td><center><img src="visualization\dmcgb2\observation\train_environments_openloop_prediction\cup_catch_EAWM.gif" alt="cartpole_swingup_EAWM.gif" style="width:384px"> </center></td>
    <td ><center><img src="visualization\dmcgb2\observation\train_environments_openloop_prediction\cup_catch_dreamerV3.gif" alt="cup_catch_EAWM.gif" style="width:384px">   </center></td>
        <td><center> <img src="visualization\dmcgb2\observation\train_environments_openloop_prediction\cup_catch_DyMoDreamer.gif" alt="walker_stand_EAWM.gif" style="width:384px">  </center></td>
    </tr>
    <tr>
    <td><center>EAWM</center></td> <td><center>DreamerV3</center></td><td><center> DyMoDreamer</center></td>
    </tr>
</table>

### Open-loop prediction in unseen test environments
<table>
    <tr>
        <td><center><img src="visualization\dmcgb2\observation\test_environments_openloop_prediction\cup_catch_EAWM.gif" alt="cartpole_swingup_EAWM.gif" style="width:384px"> </center></td>
    <td ><center><img src="visualization\dmcgb2\observation\test_environments_openloop_prediction\cup_catch_dreamerV3.gif" alt="cup_catch_EAWM.gif" style="width:384px"> </center></td>
        <td><center> <img src="visualization\dmcgb2\observation\test_environments_openloop_prediction\cup_catch_DyMoDreamer.gif" alt="walker_stand_EAWM.gif" style="width:384px"> </center></td>
    </tr>
        <tr>
    <td><center>EAWM</center></td> <td><center>DreamerV3</center></td><td><center> DyMoDreamer</center></td>
    </tr>
</table>

As demonstrated above, we observe that observation prediction in current MBRL world models generalizes poorly to unseen test environments due to limited interaction diversity in the training environments. In contrast, event prediction is inherently more tractable and interpretable, owing to the sparsity and well-defined semantic structure of events.

## 📈 Results

The folder `results` contains raw scores of world models (for each game, and for each training run).

## Download Checkpoints

The pretrained checkpoints are hosted on Hugging Face: [darwin05/EAWM](https://huggingface.co/darwin05/EAWM).

You can download the full checkpoint repository with the Hugging Face Hub Python API:

```bash
pip install -U huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="darwin05/EAWM",
    local_dir="checkpoints"
)
```

Alternatively, you can use the Hugging Face CLI:

```bash
hf download darwin05/EAWM \
  --repo-type model \
  --local-dir checkpoints
```

After downloading, the checkpoint directory should follow this structure:

```text
checkpoints/
|-- EADream/
|   |-- atari_pong.pt
|   |-- atari_breakout.pt
|   `-- dmc_cheetah_run.pt
`-- EASimulus/
    |-- Atari/
    |   |-- Pong.pt
    |   `-- Breakout.pt
    `-- craftax.pt
```

## Citation
```
@inproceedings{
    Peng2026from,
    title={From Observations to Events: Event-Aware World Models for Reinforcement Learning},
    author={Zhao-Han Peng and Shaohui Li and Zhi Li and Shulan Ruan and Yu Liu and You He},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=OWkkFaq1IZ}
}
```

## Acknowledgements

The code in this repository is built upon the following repositories.

- [https://github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- [https://github.com/leor-c/Simulus](https://github.com/leor-c/Simulus)
- [https://github.com/fkodom/yet-another-retnet](https://github.com/fkodom/yet-another-retnet)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/wandb/wandb](https://github.com/wandb/wandb)
- [https://github.com/thuml/HarmonyDream](https://github.com/thuml/HarmonyDream)
- [https://github.com/NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [https://github.com/aalmuzairee/dmcgb2](https://github.com/aalmuzairee/dmcgb2)
