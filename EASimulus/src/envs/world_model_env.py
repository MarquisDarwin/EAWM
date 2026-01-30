from collections import deque
from typing import List, Optional, Union

import gymnasium as gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torchvision

from utils import RecurrentState, ObsModality
from models.world_model import POPWorldModel
from models.tokenizer import MultiModalTokenizer


def np_obs_to_tensor(obs, device):
    return torchvision.transforms.functional.to_tensor(obs).to(device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]


def np_obs_to_tokens_tensor(obs, device, tokenizer):
    if tokenizer is None:
        return torch.from_numpy(obs).long().unsqueeze(0).to(device)
    return tokenizer.encode(np_obs_to_tensor(obs, device), should_preprocess=True).tokens


class POPWorldModelEnv:

    def __init__(
            self, tokenizer: MultiModalTokenizer, world_model: POPWorldModel,
            device: Union[str, torch.device], env: Optional[gym.Env] = None,
            real_reward_weight:float = 1.0, intrinsic_reward_weight: float = 1.0
    ) -> None:
        self.prior_context = None
        self.real_reward_weight = real_reward_weight
        self.intrinsic_reward_weight = intrinsic_reward_weight

        self.device = torch.device(device)
        self.world_model = world_model.eval()
        self.tokenizer = tokenizer.eval() if tokenizer is not None else None
        self.modality_curiosity_balance={
            ObsModality.image: 1.0,
            ObsModality.vector: 1.0,
            ObsModality.token: 1.0,
            ObsModality.token_2d: 1.0
        }
        self.recurrent_state = None
        self._tokens_per_obs = world_model.tokens_per_obs
        self.last_obs_tokens = None
        self.keys_values_wm = None
        self.env = env

    @property
    def tokens_per_obs(self) -> int:
        return self._tokens_per_obs

    @torch.no_grad()
    def reset(self) -> Tensor:
        assert self.env is not None
        # obs = self.env.reset()
        # obs_tokens = np_obs_to_tokens_tensor(obs, self.device, self.tokenizer)
        # ctx = [obs_tokens]
        # for i in range(self.world_model.context_length - 1):
        #     action = torch.zeros(1, 1, device=self.device).long()  # noop
        #     obs, reward, done, info = self.env.step(int(action[0].item()))
        #     ctx.append(action)
        #     ctx.append(np_obs_to_tokens_tensor(obs, self.device, self.tokenizer))
        # ctx = torch.cat(ctx, dim=1)
        # return self.reset_from_initial_observations(ctx)
        raise NotImplementedError()

    @torch.no_grad()
    def reset_from_initial_observations(self, ctx_tokens_emb, return_tokens: bool = False,
                                        **kwargs) -> Tensor:
        self.refresh_state_with_initial_obs_tokens(ctx_tokens_emb)
        self.last_obs_tokens_emb = ctx_tokens_emb[:, -self._tokens_per_obs:]

        return self.decode_obs_tokens() if not return_tokens else ctx_tokens_emb[:, -self._tokens_per_obs:]

    @torch.no_grad()
    def refresh_state_with_initial_obs_tokens(self, ctx_tokens_emb: Tensor) -> Tensor:
        assert ctx_tokens_emb.dim() == 3, f"Got {ctx_tokens_emb.dim()} ({ctx_tokens_emb.shape})"
        n, num_ctx_tokens = ctx_tokens_emb.shape[:2]
        action_seq_len = self.world_model.tokens_per_action
        assert (num_ctx_tokens + action_seq_len) % self.world_model.tokens_per_block == 0

        self.recurrent_state = RecurrentState(None, 0)
        self.prior_context = ctx_tokens_emb

        return ctx_tokens_emb[:, -self._tokens_per_obs:]

    def query_world_model(self, tokens_emb):
        assert tokens_emb.dim() == 3
        tokens_emb = rearrange(tokens_emb, 'b (t k1) e -> b t k1 e', k1=self.world_model.tokens_per_block)
        return self.world_model.forward_inference(tokens_emb, recurrent_state=self.recurrent_state)

    @torch.no_grad()
    def _compute_reward_and_done(self, outputs_wm: torch.Tensor):
        if self.world_model.compute_states_parallel_inference:
            k1 = self.world_model.tokens_per_block
            outputs_wm = outputs_wm[:, -k1 - 1:-k1]
            raise NotImplementedError()

        rewards, ends = self.world_model.sample_rewards_ends(outputs_wm)
        rewards = rewards.reshape(-1)  # (B,)
        ends = ends.reshape(-1)  # (B,)

        return rewards, ends

    def _embed_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.clone().detach()
        else:
            assert isinstance(action, np.ndarray)
            dtype = torch.long if np.issubdtype(action.dtype, np.integer) else torch.float
            action = torch.tensor(action, dtype=dtype, device=self.device)

        assert action.shape[1] == 1, f"Got shape {action.shape}"
        return self.world_model.embed_actions(action).flatten(1, 2)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True,
             return_tokens: bool = False):
        assert (
                (self.keys_values_wm is not None or self.recurrent_state is not None)
                and
                self.tokens_per_obs is not None
        )

        action_emb = self._embed_action(action)

        if self.prior_context is not None:
            tokens_emb = torch.cat([self.prior_context, action_emb], dim=1)
        else:
            tokens_emb = action_emb
        outputs_wm = self.query_world_model(tokens_emb)

        self.last_obs_tokens, reward, done = self._compute_next_obs_tokens(outputs_wm)
        if self.world_model.uses_pop:
            obs_tokens = {k: v.unsqueeze(1) for k, v in self.last_obs_tokens.items()}
            self.prior_context = self.world_model.embed_obs_tokens(obs_tokens, self.tokenizer).squeeze(1)
        else:
            self.prior_context = None

        obs = self.decode_obs_tokens() if not return_tokens else self.last_obs_tokens
        return obs, reward, done, None

    def _compute_next_obs_tokens(self, last_wm_output: torch.Tensor):
        if self.world_model.compute_states_parallel_inference:
            # preds = last_wm_output[:, -self.world_model.tokens_per_block:-self.world_model.tokens_per_action]
            raise NotImplementedError()
        else:
            preds = self.world_model.compute_next_obs_pred_latents(self.recurrent_state)[0]
        next_obs_tokens = self.world_model.sample_obs_tokens(preds)
        rewards, ends = self.world_model.sample_rewards_ends(preds)
        if self.world_model.enable_curiosity:
            d = self.world_model.tokens_per_obs_dict
            preds = torch.split(preds, [d[m] for m in self.world_model.ordered_modalities], dim=1)
            intrinsic_reward = torch.cat([
                self.world_model.curiosity_head[m.name].estimate_uncertainty(preds[i])[0].mean(-1, keepdim=True)*self.modality_curiosity_balance[m]
                for i, m in enumerate(self.world_model.ordered_modalities)
            ], dim=-1).sum(dim=-1, keepdim=True)
            assert rewards.shape == intrinsic_reward.shape, f"{rewards.shape}; {intrinsic_reward.shape}"
            rewards = self.real_reward_weight * rewards + self.intrinsic_reward_weight * intrinsic_reward
        return next_obs_tokens, rewards, ends
    @torch.no_grad()
    def _compute_total_reward_with_real_env(self, action: Union[int, np.ndarray, torch.LongTensor],rewards:Tensor):
        action_emb = self._embed_action(action)

        if self.prior_context is not None:
            tokens_emb = torch.cat([self.prior_context, action_emb], dim=1)
        else:
            tokens_emb = action_emb
        outputs_wm = self.query_world_model(tokens_emb)
        if self.world_model.compute_states_parallel_inference:
            # preds = last_wm_output[:, -self.world_model.tokens_per_block:-self.world_model.tokens_per_action]
            raise NotImplementedError()
        else:
            preds = self.world_model.compute_next_obs_pred_latents(self.recurrent_state)[0]
        if self.world_model.enable_curiosity:
            d = self.world_model.tokens_per_obs_dict
            preds = torch.split(preds, [d[m] for m in self.world_model.ordered_modalities], dim=1)
            intrinsic_reward = torch.cat([
                self.world_model.curiosity_head[m.name].estimate_uncertainty(preds[i])[0].mean(-1, keepdim=True)
                for i, m in enumerate(self.world_model.ordered_modalities)
            ], dim=-1).sum(dim=-1, keepdim=True)
            assert rewards.shape == intrinsic_reward.shape, f"{rewards.shape}; {intrinsic_reward.shape}"
            rewards = self.real_reward_weight * rewards + self.intrinsic_reward_weight * intrinsic_reward
        return rewards
    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> Tensor:
        embedded_tokens = self.tokenizer[ObsModality.image].embedding(self.last_obs_tokens[ObsModality.image])  # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(embedded_tokens.shape[1])))
        rec = self.tokenizer[ObsModality.image].decode(z, should_postprocess=True)  # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.last_obs_tokens[ObsModality.image].shape == (1, self.tokens_per_obs)
        return self.render_batch()[0]


class POPWMEnv4Play(POPWorldModelEnv):

    def __init__(self, tokenizer: MultiModalTokenizer, world_model: POPWorldModel, device: Union[str, torch.device],
                 env: Optional[gym.Env] = None) -> None:
        super().__init__(tokenizer, world_model, device, env)
        self.horizon = self.world_model.config.max_blocks
        self.context_length = self.world_model.context_length
        self.current_step = self.context_length
        self.next_context = deque([])

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        self.current_step = self.context_length
        self.next_context.clear()

        obs = self.env.reset()
        obs_tokens = np_obs_to_tokens_tensor(obs, self.device, self.tokenizer)
        self.next_context.append(obs_tokens)
        for i in range(self.world_model.context_length - 1):
            action = torch.zeros(1, 1, device=self.device).long()  # noop
            obs, reward, done, info = self.env.step(int(action[0].item()))
            self.next_context.append(action)
            self.next_context.append(np_obs_to_tokens_tensor(obs, self.device, self.tokenizer))
        ctx = torch.cat([v for v in self.next_context], dim=1)
        return self.reset_from_initial_observations(ctx, True)

    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True,
             return_tokens: bool = False, clip_context: bool = False) -> None:
        assert should_predict_next_obs
        res = super().step(action, should_predict_next_obs, return_tokens)
        action = action if isinstance(action, int) else action.flatten()[0]
        obs, reward, done, info = self.env.step(action)

        if clip_context:
            action_token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action,
                                                                                                         dtype=torch.long,
                                                                                                         device=self.device)
            action_token = action_token.reshape(-1, 1)  # (B, 1)
            obs_tokens = np_obs_to_tokens_tensor(obs, self.device, self.tokenizer)
            self.next_context.append(action_token)
            self.next_context.append(obs_tokens)

            while len(self.next_context) > self.context_length * 2 - 1:
                self.next_context.popleft()

            self.current_step += 1

            if self.current_step % self.horizon - self.context_length == 0:
                self.refresh_state_with_initial_obs_tokens(torch.cat([v for v in self.next_context], dim=1))

        return res[0], res[1], done, info



