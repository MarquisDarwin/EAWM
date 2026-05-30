import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import numpy as np
import torch

try:
    import ruamel.yaml as yaml
except ModuleNotFoundError:
    import yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools

DEFAULT_CONFIG = (
    "logno1harfocal_15ga4adamwa_5m_5dizzyattr_05dyn_5stoch6432u512mc32#adamw/configsc.yaml"
)
DEFAULT_RESULT_FILE = "log/result.txt"


def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _load_config(config_path, config_names):
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        config_path = pathlib.Path(__file__).parent / config_path
    configs = yaml.safe_load(config_path.read_text())

    defaults = {}
    for name in ["defaults", *config_names]:
        _recursive_update(defaults, configs[name])
    defaults = {key: tools.args_type(value)(value) for key, value in defaults.items()}
    return argparse.Namespace(**defaults)


def _normalize_legacy_config(config):
    if not hasattr(config, "dizzy"):
        config.dizzy = bool(config.decoder.pop("dizzy", False))
    else:
        config.decoder.pop("dizzy", None)
    if not hasattr(config, "image_threshold"):
        config.image_threshold = 16
    if not hasattr(config, "stackframe"):
        config.stackframe = False
    if not hasattr(config, "mha"):
        config.mha = False
    if not hasattr(config, "mha_layer"):
        config.mha_layer = {"layers": 1, "dropout": 0.0, "num_heads": 2}
    if not hasattr(config, "mae_ratio"):
        config.mae_ratio = 1.0
    return config


def _normalize_key(key):
    return key.replace("._orig_mod.", ".")


def _is_eval_key(key):
    return (
        key.startswith("_wm.encoder.")
        or key.startswith("_wm.dynamics.")
        or key.startswith("_task_behavior.actor.")
        or key.startswith("_task_behavior.value.")
        or key.startswith("_task_behavior._slow_value.")
        or key == "_task_behavior.ema_vals"
        or key.startswith("tokenizer.")
        or key.startswith("actor_critic.")
    )


def _load_eval_state_dict(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    metadata = {
        key: checkpoint[key]
        for key in ("task", "game", "seed", "step", "eval_return", "source_run")
        if isinstance(checkpoint, dict) and key in checkpoint
    }

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        raw_state = checkpoint["agent_state_dict"]
    else:
        raw_state = checkpoint

    state_dict = {}
    for key, value in raw_state.items():
        key = _normalize_key(key)
        if _is_eval_key(key):
            state_dict[key] = value
    return state_dict, metadata


class EvalAgent(torch.nn.Module):
    def __init__(self, obs_space, act_space, config):
        super().__init__()
        self._config = config
        self._wm = models.WorldModel(obs_space, act_space, 0, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)

    def forward(self, obs, reset, state=None, training=False):
        if training:
            raise ValueError("eval.py only supports evaluation mode")
        if state is None:
            latent = action = None
        else:
            latent, action = state

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs, batch=False)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        actor = self._task_behavior.actor(feat)
        action = actor.mode()
        logprob = actor.log_prob(action)
        latent = {key: value.detach() for key, value in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        return {"action": action, "logprob": logprob}, (latent, action)


class AtariTimeLimit:
    def __init__(self, env, duration):
        self.env = env
        self._duration = duration
        self._step = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self._step = 0
        return self.env.reset()

    def step(self, action):
        if self._step is None:
            raise RuntimeError("Must reset environment before stepping.")
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info.setdefault("discount", np.array(1.0).astype(np.float32))
            self._step = None
        return obs, reward, done, info

    def close(self):
        return self.env.close()


def _make_atari_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite != "atari":
        raise NotImplementedError(suite)
    import envs.atari as atari

    env = atari.Atari(
        task,
        config.action_repeat,
        config.size,
        gray=config.grayscale,
        noops=config.noops,
        lives=config.lives,
        sticky=config.stickey,
        actions=config.actions,
        resize=config.resize,
        seed=config.seed + id,
    )
    return AtariTimeLimit(env, config.time_limit)


def _make_dmc_env(config, id):
    import envs.dmc as dmc
    import envs.wrappers as wrappers

    _, task = config.task.split("_", 1)
    env = dmc.DeepMindControl(
        task, config.action_repeat, config.size, seed=config.seed + id
    )
    env = wrappers.NormalizeActions(env)
    return wrappers.TimeLimit(env, config.time_limit)


def _make_dmcgb_env(config, id, gb_mode="color_hard"):
    import envs.dmcgb as dmcgb
    import envs.wrappers as wrappers

    _, task = config.task.split("_", 1)
    env = dmcgb.DeepMindControl(
        task,
        config.action_repeat,
        config.size,
        seed=config.seed + id,
        max_episode_steps=(config.time_limit + config.action_repeat - 1)
        // config.action_repeat,
    )
    assert gb_mode in wrappers.VALID_MODES, f'Unsupported dmcgb mode "{gb_mode}"'
    env = wrappers.ShiftWrapper(env, gb_mode, config.seed)
    env = wrappers.RotateWrapper(env, gb_mode, config.seed)
    env = wrappers.ColorVideoWrapper(env, gb_mode, config.seed, video_render_size=64)
    env._domain_name = "dmcgb"
    env = wrappers.NormalizeActions(env)
    return wrappers.TimeLimit(env, config.time_limit)


def _make_eval_env(config, id):
    suite, _ = config.task.split("_", 1)
    if suite == "atari":
        return _make_atari_env(config, id)
    if suite == "dmc":
        return _make_dmc_env(config, id)
    if suite == "dmcgb":
        return _make_dmcgb_env(config, id)
    raise NotImplementedError(suite)


def _make_agent(config, env):
    agent = EvalAgent(env.observation_space, env.action_space, config).to(config.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()
    return agent


def _prepare_obs(obs, background, kernel, stackframe=False, previous_image=None):
    obs = obs.copy()
    if previous_image is None:
        obs["event"] = np.zeros(obs["image"].shape[:2], dtype=np.uint8)
        background.apply(obs["image"])
        if stackframe:
            obs["image"] = np.tile(obs["image"], (3, 1, 1, 1))
    else:
        mask = background.apply(obs["image"])
        obs["event"] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if stackframe:
            obs["image"] = np.concatenate(
                (previous_image[1:], obs["image"][None]), axis=0
            )
    return obs


def evaluate(agent, env, config, episodes):
    scores = []
    lengths = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while len(scores) < episodes:
        background = cv2.createBackgroundSubtractorMOG2(
            varThreshold=config.image_threshold
        )
        obs = _prepare_obs(env.reset(), background, kernel, config.stackframe)
        state = None
        score = 0.0
        length = 0
        done = False

        while not done:
            batched = {
                key: np.stack([value])
                for key, value in obs.items()
                if "log_" not in key
            }
            with torch.inference_mode():
                policy_output, state = agent(
                    batched, np.array([length == 0]), state, training=False
                )
            action = {
                key: np.array(value[0].detach().cpu())
                for key, value in policy_output.items()
            }
            next_obs, reward, done, _ = env.step(action["action"])
            previous_image = obs["image"] if config.stackframe else None
            obs = _prepare_obs(
                next_obs,
                background,
                kernel,
                config.stackframe,
                previous_image,
            )
            score += float(reward)
            length += 1

        scores.append(score)
        lengths.append(length)
        print(
            f"episode={len(scores)}/{episodes} "
            f"return={score:.2f} mean_return={np.mean(scores):.2f}"
        )

    return float(np.mean(scores)), float(np.mean(lengths))


def _close_env(env):
    try:
        env.close()
        return
    except AttributeError:
        pass

    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    close = getattr(getattr(inner, "_env", None), "close", None)
    if callable(close):
        close()


def _infer_task(weights_path, metadata, explicit_task, explicit_game):
    if explicit_task:
        return explicit_task
    if "task" in metadata:
        return metadata["task"]
    if explicit_game:
        if explicit_game.startswith(("atari_", "dmc_", "dmcgb_")):
            return explicit_game
        return f"atari_{explicit_game}"

    name = str(metadata.get("game") or pathlib.Path(weights_path).stem)
    if name.startswith(("atari_", "dmc_", "dmcgb_")):
        return name
    return f"atari_{name}"


def _default_configs_for_task(task):
    if task.startswith("atari_"):
        return ["atari100k"]
    if task.startswith(("dmc_", "dmcgb_")):
        return ["dmc_vision"]
    return []


def _task_to_game(task):
    if task.startswith("atari_"):
        return task.split("_", 1)[1]
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--game", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--configdir", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result-file", type=pathlib.Path, default=DEFAULT_RESULT_FILE)
    args = parser.parse_args()

    state_dict, metadata = _load_eval_state_dict(args.weights)
    if not state_dict:
        raise RuntimeError(f"No eval state_dict keys found in {args.weights}")

    task = _infer_task(args.weights, metadata, args.task, args.game)
    game = _task_to_game(task)
    seed = args.seed if args.seed is not None else int(metadata.get("seed", 0))
    config_names = args.configs or _default_configs_for_task(task)
    config = _normalize_legacy_config(_load_config(args.configdir, config_names))
    config.task = task
    config.seed = seed
    config.compile = False
    config.video_pred_log = False
    config.eval_episode_num = args.episodes or config.eval_episode_num
    config.device = args.device
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        config.device = "cpu"

    tools.set_seed_everywhere(config.seed)
    config.steps = int(float(config.steps)) // config.action_repeat
    config.eval_every = int(float(config.eval_every)) // config.action_repeat
    config.log_every = int(float(config.log_every)) // config.action_repeat
    config.time_limit = int(float(config.time_limit)) // config.action_repeat

    env = _make_eval_env(config, 0)
    config.num_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )
    agent = _make_agent(config, env)
    _missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    required_groups = {
        "_wm.encoder": any(key.startswith("_wm.encoder.") for key in state_dict),
        "_wm.dynamics": any(key.startswith("_wm.dynamics.") for key in state_dict),
        "_task_behavior.actor": any(
            key.startswith("_task_behavior.actor.") for key in state_dict
        ),
    }
    missing_groups = [name for name, present in required_groups.items() if not present]
    if missing_groups:
        raise RuntimeError(f"Missing required weight groups: {missing_groups}")
    if unexpected:
        print(f"Ignored unexpected keys: {len(unexpected)}")

    try:
        eval_return, eval_length = evaluate(agent, env, config, config.eval_episode_num)
    finally:
        _close_env(env)

    result = {
        "game": game,
        "task": task,
        "weights": str(args.weights),
        "seed": config.seed,
        "episodes": config.eval_episode_num,
        "eval_return": eval_return,
        "eval_length": eval_length,
    }
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    with args.result_file.open("a") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
