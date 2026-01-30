from functools import partial
from pathlib import Path

import click
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
from scipy.ndimage import rotate

from main import build_agent
from envs import SingleProcessEnv, POPWMEnv4Play
from game import AgentEnv, CraftaxAgentEnv, EpisodeReplayEnv, Game
from utils.preprocessing import get_obs_processor


def play_atari(cfg: DictConfig, mode, reconstruction_mode, header_info, fps, model_path: Path):
    save_mode = 0

    device = torch.device(cfg.common.device)
    assert mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    if mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]
    print(f"size: {size}")

    if mode == 'episode_replay':
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'))
        keymap = 'episode_replay'

    else:
        agent = build_agent(test_env, cfg, device)
        if model_path is not None:
            agent.load(model_path, device)

        if mode == 'play_in_world_model':
            env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                env=env_fn())
            keymap = cfg.env.keymap

        elif mode == 'agent_in_env':
            env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=reconstruction_mode)
            keymap = 'empty'
            if reconstruction_mode:
                size[1] *= 3

        elif mode == 'agent_in_world_model':
            wm_env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                   env=env_fn())
            env = AgentEnv(agent, wm_env, cfg.env.keymap, do_reconstruction=False)
            keymap = 'empty'

    game = Game(env, keymap_name=keymap, size=size, fps=fps, verbose=bool(header_info),
                record_mode=bool(save_mode))
    game.run()


def play_craftax(cfg: DictConfig, fps: int, actions_info: bool, model_path: Path):
    device = torch.device(cfg.common.device)
    from envs.wrappers.craftax import make_craftax
    env = SingleProcessEnv(make_craftax)

    # Determine the screen size:
    from craftax.craftax.play_craftax import BLOCK_PIXEL_SIZE_HUMAN, OBS_DIM, INVENTORY_OBS_HEIGHT
    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN
    width = OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size
    height = (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size
    screen_size = (height, width)

    # Initialize the agent:
    agent = build_agent(env, cfg, device=device)
    if model_path is not None:
        agent.load(model_path, device, load_tokenizer=False, load_world_model=True, load_actor_critic=True)

    # Set up the game wrappers:
    env = CraftaxAgentEnv(agent, env, pixel_render_size=pixel_render_size)

    game = Game(env, keymap_name='empty', size=screen_size, fps=fps, verbose=True, record_mode=False)
    game.run()


def get_config(benchmark: str):
    from hydra import compose, initialize
    initialize(version_base=None, config_path="../config", job_name="play")
    overrides = ['hydra.run.dir=.', 'hydra.output_subdir=null']
    should_override_benchmark = Path('config/benchmark').exists()
    if should_override_benchmark:
        overrides.append(f"benchmark={benchmark}")
    cfg = compose(config_name="base", overrides=overrides)
    return cfg


@click.command()
@click.option('-m', '--mode',
              type=click.Choice(['episode_replay', 'agent_in_env', 'agent_in_world_model', 'agent_in_world_model']),
              default='agent_in_env')
@click.option('-r', '--reconstruction-mode', is_flag=True, show_default=True, default=False, help='Reconstruction mode. Shows the original observation (left), downscaled observation (center), and reconstructed obs (right) - how the agent sees the world.')
@click.option('-h', '--header-info', is_flag=True, show_default=True, default=False, help='Show cumulative return, controller actions, and step info.')
@click.option('--fps', type=click.IntRange(min=1, max=240), default=15, help='frames per second')
@click.option('-p', '--model-path', type=click.Path(exists=True))
def atari(mode, reconstruction_mode, header_info, fps, model_path):
    cfg = get_config(benchmark='atari')
    play_atari(cfg, mode, reconstruction_mode, header_info, fps, model_path)


@click.command()
@click.option('-m', '--mode', type=click.Choice(['agent_in_env']), default='agent_in_env')
@click.option('--fps', type=click.IntRange(min=1, max=240), default=15, help='frames per second')
@click.option('-i', '--actions-info', is_flag=True, show_default=True, default=False, help='Print the actions of the agent.')
@click.option('-p', '--model-path', type=click.Path(exists=True))
def craftax(mode, fps, actions_info, model_path):
    cfg = get_config(benchmark='craftax')
    play_craftax(cfg, fps, actions_info, model_path)


@click.group()
def main():
    pass


main.add_command(atari)
main.add_command(craftax)


if __name__ == "__main__":
    main()
