from abc import abstractmethod, ABC
from math import ceil
from typing import Any, Optional
import sys

from loguru import logger
from einops import rearrange
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from yet_another_retnet.retnet import RetNetDecoder, RetNetDecoderLayer
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import POPWorldModelEnv
from models.actor_critic.encoders import ObsEncoderBase
from models.actor_critic.types import *
from models.tokenizer import MultiModalTokenizer
from models.world_model import POPWorldModel
from models.embedding import make_mlp, MultiDiscreteEmbedding
from utils import (
    compute_lambda_returns,
    LossWithIntermediateLosses,
    QuantizedContinuousDistribution,
    LSTMCellWrapper,
    ObsModality,
    HLGaussCategoricalRegressionHead,
)
from utils.types import MultiModalObs
from utils.preprocessing import BufferScaler
from utils.distributions import (
    CategoricalDistribution,
    SquashedDiagNormalDistribution,
    MultiCategoricalDistribution,
)


class ActorCriticLS(nn.Module):
    """
    This version works in latent space. it receives the token codes of the observation frame (embed_dim, w, h)
    and applies a CNN + dense to map it to a latent vector which is the input to the LSTM.
    Importantly, the token codes are the learned codes of the tokenizer.
    """

    def __init__(
        self,
        obs_encoders: dict[ObsModality, ObsEncoderBase],
        obs_mlp_layer_sizes: list[int],
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_encoders = nn.ModuleDict({k.name: v for k, v in obs_encoders.items()})
        self.separate_networks = separate_networks
        self.obs_latent_fuser = make_mlp(
            layer_dims=[sum([enc.out_dim for enc in self.obs_encoders.values()])]
            + obs_mlp_layer_sizes,
            linear_out=False,
            device=device,
        )
        self._ordered_modalities = [
            modality for modality in ObsModality if modality.name in self.obs_encoders
        ]
        self.device = device
        self.name = name
        self.include_action_inputs = include_action_inputs
        self.context_len = context_len
        self.real_reward_weight = real_reward_weight
        self.intrinsic_reward_weight = intrinsic_reward_weight

        self.lstm_dim = obs_mlp_layer_sizes[-1]
        self.embed_dim = obs_mlp_layer_sizes[-1]

        self.rnn_type = rnn_type
        self.actor, self.actor_state = self._build_rnn_model()

        if self.separate_networks:
            self.critic_encoders = nn.ModuleDict(
                {
                    k.name: v.build_another() if self.separate_networks else None
                    for k, v in obs_encoders.items()
                }
            )
            self.critic_latent_fuser = make_mlp(
                layer_dims=[sum([enc.out_dim for enc in self.obs_encoders.values()])]
                + obs_mlp_layer_sizes,
                linear_out=False,
                device=device,
            )
            self.critic, self.critic_state = self._build_rnn_model()
        else:
            self.critic_encoders = {k.name: None for k, v in obs_encoders.items()}
            self.critic_latent_fuser = None
            self.critic, self.critic_state = None, None

        self.critic_v_head = self._build_critic_head()
        self.actor_linear = self._build_actor_head()

        # self.action_emb_map = nn.Linear(token_embed_dim, lstm_latent_dim) if token_embed_dim != lstm_latent_dim else None
        self.action_emb_map = None

        self.return_scaler = BufferScaler()

        logger.info(
            f"Initialized ActorCriticLS (separate networks: {separate_networks})"
        )
        logger.info(
            f"reward weights: real: {self.real_reward_weight}, intrinsic: {self.intrinsic_reward_weight}"
        )

    def _build_rnn_model(self):
        if self.rnn_type == "lstm":
            lstm = nn.LSTM(
                self.embed_dim,
                self.lstm_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                device=self.device,
            )
            lstm = LSTMCellWrapper(lstm)
            # lstm = nn.LSTMCell(self.lstm_dim, self.lstm_dim, device=self.device)
            lstm_state = (None, None)
            return lstm, lstm_state

        elif self.rnn_type == "gru":
            gru = nn.GRUCell(self.embed_dim, self.lstm_dim, device=self.device)
            gru_state = None
            return gru, gru_state

    @abstractmethod
    def _build_actor_head(self) -> nn.Module:
        pass

    def _build_critic_head(self) -> nn.Module:
        return nn.Sequential(
            # nn.LayerNorm(self.lstm_dim),
            nn.Linear(self.lstm_dim, 1, device=self.device)
        )

    def __repr__(self) -> str:
        return self.name

    def clear(self) -> None:
        if self.rnn_type == "lstm":
            self.actor_state = (None, None)
        elif self.rnn_type == "gru":
            self.actor_state = None

    def get_zero_rnn_state(self, n, device, rnn_type: str = None):
        if rnn_type is None:
            rnn_type = self.rnn_type

        if rnn_type == "lstm":
            return torch.zeros(n, self.lstm_dim, device=device), torch.zeros(
                n, self.lstm_dim, device=device
            )
        elif rnn_type == "gru":
            return torch.zeros(n, self.lstm_dim, device=device)
        else:
            assert False, f"rnn type '{rnn_type}' not supported"

    def embed_obs(self, observation: MultiModalObs) -> tuple[Tensor, Tensor]:
        assert set([k.name for k in observation.keys()]) == set(
            self.obs_encoders.keys()
        ), f"{set(observation.keys())} != {set(self.obs_encoders.keys())}"

        actor_emb = torch.cat(
            [
                self.obs_encoders[k.name](observation[k])
                for k in self._ordered_modalities
            ],
            dim=-1,
        )
        actor_emb = self.obs_latent_fuser(actor_emb)
        critic_emb = None
        if self.separate_networks:
            critic_emb = torch.cat(
                [
                    self.critic_encoders[k.name](observation[k])
                    for k in self._ordered_modalities
                ],
                dim=-1,
            )
            critic_emb = self.critic_latent_fuser(critic_emb)

        return actor_emb, critic_emb

    @abstractmethod
    def embed_action(self, action) -> tuple[Tensor, Tensor]:
        pass

    def process_action(self, action: Tensor, mask_padding: Tensor = None):
        assert self.include_action_inputs
        assert action is not None

        x = self.embed_action(action)
        return self.process_action_emb(x, mask_padding)

    def process_action_emb(
        self, ac_action_embs: tuple[Tensor, Tensor], mask_padding: Tensor = None
    ):
        assert (
            self.include_action_inputs
            and ac_action_embs[0].dim() == 2
            and (not self.separate_networks or ac_action_embs[1].dim() == 2),
            f"Got {ac_action_embs[0].dim()} ({ac_action_embs[0].shape})",
        )
        self.actor_state = self._rnn_forward(
            ac_action_embs[0], mask_padding, self.actor, self.actor_state
        )
        self.critic_state = self._rnn_forward(
            ac_action_embs[1], mask_padding, self.critic, self.critic_state
        )

    def reset(
        self,
        n: int,
        burnin_observations: Optional[MultiModalObs] = None,
        mask_padding: Optional[Tensor] = None,
        ac_actions_embs: tuple[Tensor, Tensor] = None,
    ) -> None:
        assert ac_actions_embs is None or (
            ac_actions_embs[0].dim() == 3
            and (not self.separate_networks or ac_actions_embs[1].dim() == 3)
        )  # (b, t, e)

        device = self.device
        self.actor_state = self.get_zero_rnn_state(n, device)
        self.critic_state = (
            self.get_zero_rnn_state(n, device) if self.separate_networks else None
        )
        if burnin_observations is not None:
            batch_dims = set(v.shape[:2] for v in burnin_observations.values())
            assert len(batch_dims) == 1
            batch_dims = batch_dims.pop()
            assert (
                batch_dims[0] == n
                and mask_padding is not None
                and batch_dims == mask_padding.shape
            )
            assert batch_dims == ac_actions_embs[0].shape[:2]
            for i in range(batch_dims[1]):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(
                            {k: v[:, i] for k, v in burnin_observations.items()},
                            mask_padding[:, i],
                        )
                        # if self.include_action_inputs:
                        #     cur_actions_embs = (ac_actions_embs[0][:, i],
                        #                         ac_actions_embs[1][:, i] if ac_actions_embs[1] is not None else None)
                        #     self.process_action_emb(ac_action_embs=cur_actions_embs, mask_padding=mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        if self.rnn_type == "lstm":
            hx, cx = self.actor_state
            hx = hx[mask]
            cx = cx[mask]
            self.actor_state = (hx, cx)
        elif self.rnn_type == "gru":
            self.actor_state = self.actor_state[mask]

    def _rnn_forward(self, x, mask_padding, model, rnn_state):
        if model is None:
            # Shared actor-critic network
            return None

        if mask_padding is None:
            rnn_state = model(x, rnn_state)
        else:
            if self.rnn_type == "lstm":
                hx, cx = rnn_state
                hx[mask_padding], cx[mask_padding] = model(
                    x[mask_padding], (hx[mask_padding], cx[mask_padding])
                )
                rnn_state = (hx, cx)

            elif self.rnn_type == "gru":
                rnn_state[mask_padding] = model(x, rnn_state[mask_padding])
            else:
                assert False, f"rnn type '{self.rnn_type}' not supported"
        return rnn_state

    def rnn_forward(self, x, mask_padding):
        self.actor_state = self._rnn_forward(
            x, mask_padding, self.actor, self.actor_state
        )
        self.critic_state = self._rnn_forward(
            x, mask_padding, self.critic, self.critic_state
        )

    def get_rnn_output(self):
        if self.rnn_type == "lstm":
            return self.actor_state[0]
        elif self.rnn_type == "gru":
            return self.actor_state
        else:
            assert False, f"rnn type '{self.rnn_type}' not supported"

    def get_critic_rnn_output(self):
        if not self.separate_networks:
            return self.get_rnn_output()

        if self.rnn_type == "lstm":
            return self.critic_state[0]
        elif self.rnn_type == "gru":
            return self.critic_state
        else:
            assert False, f"rnn type '{self.rnn_type}' not supported"

    def forward(
        self, inputs: MultiModalObs, mask_padding: Optional[torch.BoolTensor] = None
    ) -> tuple[ActorOutput, CriticOutput]:
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.any())
        x_actor, x_critic = self.embed_obs(inputs)  # (b, d_lstm)
        self.actor_state = self._rnn_forward(
            x_actor, mask_padding, self.actor, self.actor_state
        )
        self.critic_state = self._rnn_forward(
            x_critic, mask_padding, self.critic, self.critic_state
        )

        return (
            self._compute_actor_output(self.get_rnn_output()),
            self._compute_critic_output(self.get_critic_rnn_output()),
        )

    @abstractmethod
    def _compute_actor_output(self, actor_latent) -> ActorOutput:
        pass

    @abstractmethod
    def _compute_critic_output(self, critic_latent) -> CriticOutput:
        pass

    def _get_actions_distribution(self, outputs: ImagineOutput):
        pass

    def _get_values_means(self, values_info: ValuesInfo) -> Tensor:
        return values_info.value_means

    def compute_loss(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        imagine_horizon: int,
        gamma: float,
        lambda_: float,
        entropy_weight: float,
        epoch: int,
        actor_start_epoch: int,
        imagine: bool,
        **kwargs: Any,
    ) -> tuple[LossWithIntermediateLosses, dict]:
        if imagine:
            outputs = self.imagine(
                batch, tokenizer, world_model, horizon=imagine_horizon
            )
            info_prefix = "imagined_"
        else:
            outputs = self.get_action_distribution_and_values(
                batch, tokenizer, world_model, horizon=imagine_horizon
            )
            info_prefix = "real_"
        # outputs = self.play_env(batch, tokenizer, world_model, horizon=imagine_horizon)

        values_means = self._get_values_means(outputs.values_info)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=values_means,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]
        self.return_scaler.update(lambda_returns)
        returns_scale = torch.maximum(
            torch.ones_like(self.return_scaler.scale), self.return_scaler.scale * 0.5
        )

        values = values_means[:, :-1]

        d = outputs.actions_distributions
        log_probs = d.log_prob(outputs.actions)[:, :-1]
        advantage = (lambda_returns - values).detach() / returns_scale
        loss_actions = -(log_probs * advantage.detach()).mean()

        loss_actor = loss_actions
        loss_entropy = -entropy_weight * d.entropy().mean()
        if epoch < actor_start_epoch:
            loss_actor = torch.zeros_like(loss_actor)
            loss_entropy = torch.zeros_like(loss_entropy)

        loss_values = self._compute_critic_loss(outputs.values_info, lambda_returns)

        info = {
            info_prefix + "rewards": outputs.rewards.detach().clone(),
            info_prefix + "returns": lambda_returns.detach().clone(),
            info_prefix + "values": values.detach().clone(),
            info_prefix + "normalized_advantage": advantage.detach().clone(),
            info_prefix + "log_probs": log_probs.detach().clone(),
            info_prefix + "returns_scale": returns_scale.item(),
        }
        if imagine:
            intermediatelosses = LossWithIntermediateLosses(
                imagined_loss_actor=loss_actor,
                imagined_loss_values=loss_values,
                imagined_loss_entropy=loss_entropy,
            )
        else:
            intermediatelosses = LossWithIntermediateLosses(
                real_loss_actor=loss_actor,
                real_loss_values=loss_values,
                real_loss_entropy=loss_entropy,
            )
        return intermediatelosses, info

    def _compute_critic_loss(
        self, values_info: ContinuousValuesInfo, targets
    ) -> Tensor:
        values = values_info.value_means
        return F.mse_loss(values[:, :-1], targets)

    def _imagination_set_initial_state(
        self, batch: Batch, tokenizer: MultiModalTokenizer, world_model: POPWorldModel
    ) -> tuple[POPWorldModelEnv, MultiModalObs]:
        device = batch["mask_padding"].device

        wm_env = POPWorldModelEnv(
            tokenizer,
            world_model,
            device=device,
            real_reward_weight=self.real_reward_weight,
            intrinsic_reward_weight=self.intrinsic_reward_weight,
        )

        batch_size, seq_len = batch["mask_padding"].shape[:2]
        mask_padding = batch["mask_padding"]

        # set the initial state of the actor-critic:
        with torch.no_grad():
            obs_tokens = world_model.get_obs_tokens(
                batch["observations"], tokenizer=tokenizer
            )
        obs_quantized = self._to_codes(obs_tokens, world_model, tokenizer)

        # Ignore last obs as it could be the last obs (obtained after termination signal)
        burnin_observations = (
            {k: v[:, :-1] for k, v in obs_quantized.items()} if seq_len > 1 else None
        )
        ac_actions_embs = self.embed_action(batch["actions"][:, :-1])
        self.reset(
            n=batch_size,
            burnin_observations=burnin_observations,
            mask_padding=mask_padding[:, :-1],
            ac_actions_embs=ac_actions_embs,
        )

        # reset WM env:
        ctx_len = self.context_len
        action_seq_len = world_model.tokens_per_action
        ctx = world_model.get_tokens_emb(
            {k: obs_tokens[k][:, -ctx_len - 1 : -1] for k in obs_tokens.keys()},
            batch["actions"][:, -ctx_len - 1 : -1],
            tokenizer=tokenizer,
        ).flatten(1, 2)[:, :-action_seq_len]

        wm_env.reset_from_initial_observations(
            ctx,
            return_tokens=True,
        )

        return wm_env, {k: v[:, -1] for k, v in obs_tokens.items()}

    @abstractmethod
    def _concat_distributions(
        self, all_actions_dists
    ) -> torch.distributions.Distribution:
        pass

    def _concat_values(
        self, values_info_list: list[ContinuousValuesInfo]
    ) -> ContinuousValuesInfo:
        values_means = [v.values for v in values_info_list]
        return ContinuousValuesInfo(
            rearrange(torch.stack(values_means, dim=1), "b t 1 -> b t")
        )

    def _to_codes(
        self,
        obs_tokens: MultiModalObs,
        world_model: POPWorldModel,
        tokenizer: MultiModalTokenizer,
    ) -> MultiModalObs:
        codes = tokenizer.to_codes(obs_tokens)
        if ObsModality.vector in codes:
            codes[ObsModality.vector] = tokenizer.tokenizers[
                ObsModality.vector.name
            ].decode(codes[ObsModality.vector])
        return codes

    def imagine(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        show_pbar: bool = False,
    ) -> ImagineOutput:
        mask_padding, actions = batch["mask_padding"], batch["actions"]

        assert mask_padding[:, -1].all()
        device = self.device

        all_actions = []
        all_actions_dists = []
        all_values_info = []
        all_q_info = []
        all_rewards = []
        all_ends = []
        all_observations = []

        wm_env, obs_tokens = self._imagination_set_initial_state(
            batch, tokenizer, world_model
        )

        obs_codes = self._to_codes(obs_tokens, world_model, tokenizer)

        effective_horizon = horizon - self.context_len + 1
        for k in tqdm(
            range(effective_horizon),
            disable=not show_pbar,
            desc="Imagination",
            file=sys.stdout,
        ):

            all_observations.append(obs_codes)

            actor_outs, critic_outs = self(inputs=obs_codes)
            action = actor_outs.get_actions_distributions().sample()
            assert self.include_action_inputs
            q = self.process_action(action.squeeze(1))
            should_predict_next_obs = k < effective_horizon - 1
            obs_tokens, reward, done, _ = wm_env.step(
                action,
                should_predict_next_obs=should_predict_next_obs,
                return_tokens=True,
            )
            obs_codes = (
                self._to_codes(obs_tokens, world_model, tokenizer)
                if should_predict_next_obs
                else None
            )

            all_actions.append(action)
            all_actions_dists.append(actor_outs.get_actions_distributions())
            all_values_info.append(critic_outs.get_value_info())
            all_rewards.append(reward.reshape(-1, 1))
            all_ends.append(done.reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations={
                k: torch.stack([o[k] for o in all_observations], dim=1)
                for k in self._ordered_modalities
            },  # (B, T, C, H, W)
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_distributions=self._concat_distributions(
                all_actions_dists
            ),  # (B, T, #actions)
            values_info=self._concat_values(all_values_info),  # (B, T)
            q_values_info=None,  # self._concat_values(all_q_info),
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),  # (B, T)
        )

    def play_env(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        show_pbar: bool = False,
    ) -> ImagineOutput:
        # initial_observations = batch['observations']
        mask_padding = batch["mask_padding"]
        # assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = next(iter(batch.values())).device
        from envs import make_dm_control

        env = make_dm_control("walker-run")

        all_actions = []
        all_actions_dists = []
        all_values_info = []
        all_q_info = []
        all_rewards = []
        all_ends = []
        all_observations = []

        self.reset(n=1)
        obs = torch.tensor([env.reset()[0]]).float().to(device)

        obs_codes = self._to_codes(tokenizer.encode(obs).tokens, world_model, tokenizer)

        # effective_horizon = horizon - self.context_len + 1
        effective_horizon = 500
        for k in tqdm(
            range(effective_horizon),
            disable=not show_pbar,
            desc="Imagination",
            file=sys.stdout,
        ):

            all_observations.append(obs_codes)

            actor_outs, critic_outs = self(inputs=obs_codes)
            action = actor_outs.get_actions_distributions().sample()
            if self.include_action_inputs:
                q = self.process_action(action.squeeze(1))
            obs, reward, terminated, truncated, _ = env.step(
                action[0].detach().cpu().numpy()
            )
            obs = torch.Tensor([obs]).float().to(action.device)
            if terminated:
                obs = torch.Tensor([env.reset()[0]]).float().to(device)
            reward = torch.Tensor([reward]).float().to(device)
            terminated = torch.Tensor([terminated]).bool().to(device)
            obs_codes = self._to_codes(
                tokenizer.encode(obs).tokens, world_model, tokenizer
            )

            all_actions.append(action)
            all_actions_dists.append(actor_outs.get_actions_distributions())
            all_values_info.append(critic_outs.get_value_info())
            # all_q_info.append(self._get_q_info(q))
            all_rewards.append(reward.reshape(-1, 1))
            all_ends.append(terminated.reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1),  # (B, T, C, H, W)
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_distributions=self._concat_distributions(
                all_actions_dists
            ),  # (B, T, #actions)
            values_info=self._concat_values(all_values_info),  # (B, T)
            q_values_info=None,  # self._concat_values(all_q_info),
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),  # (B, T)
        )

    def get_action_distribution_and_values(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        show_pbar: bool = False,
    ):
        device=batch["mask_padding"].device
        wm_env = POPWorldModelEnv(
            tokenizer,
            world_model,
            device=device,
            real_reward_weight=self.real_reward_weight,
            intrinsic_reward_weight=self.intrinsic_reward_weight,
        )

        mask_padding, actions, rewards, ends = (
            batch["mask_padding"],
            batch["actions"],
            batch["rewards"],
            batch["ends"],
        )
        batch_size, seq_len = batch["mask_padding"].shape[:2]
        # set the initial state of the actor-critic:
        with torch.no_grad():
            obs_tokens = world_model.get_obs_tokens(
                batch["observations"], tokenizer=tokenizer
            )
        obs_quantized = self._to_codes(obs_tokens, world_model, tokenizer)
        effective_horizon = horizon - self.context_len + 1
        # Ignore last obs as it could be the last obs (obtained after termination signal)
        burnin_observations = (
            {k: v[:, : -effective_horizon - 1] for k, v in obs_quantized.items()}
            if seq_len > 1
            else None
        )
        ac_actions_embs = self.embed_action(
            batch["actions"][:, : -effective_horizon - 1]
        )
        self.reset(
            n=batch_size,
            burnin_observations=burnin_observations,
            mask_padding=mask_padding[:, : -effective_horizon - 1],
            ac_actions_embs=ac_actions_embs,
        )
        ctx_len = self.context_len
        action_seq_len = world_model.tokens_per_action
        ctx = world_model.get_tokens_emb(
            {k: obs_tokens[k][:, -ctx_len - effective_horizon - 1 : - effective_horizon - 1] for k in obs_tokens.keys()},
            batch["actions"][:, -ctx_len - effective_horizon - 1 : - effective_horizon - 1],
            tokenizer=tokenizer,
        ).flatten(1, 2)[:, :-action_seq_len]

        wm_env.reset_from_initial_observations(
            ctx,
            return_tokens=True,
        )
        all_actions_dists = []
        all_values_info = []
        all_observations = []
        all_rewards=[]
        for i in tqdm(
            range(seq_len - effective_horizon - 1, seq_len - 1),
            disable=not show_pbar,
            desc="RealExperience",
            file=sys.stdout,
        ):
            total_rewards=wm_env._compute_total_reward_with_real_env(batch["actions"][:,i],batch["rewards"][:,i])
            obs_codes = {k: v[:, i] for k, v in obs_quantized.items()}
            all_observations.append(obs_codes)
            actor_outs, critic_outs = self(inputs=obs_codes)
            assert self.include_action_inputs
            all_actions_dists.append(actor_outs.get_actions_distributions())
            all_values_info.append(critic_outs.get_value_info())
            all_rewards.append(total_rewards.reshape(-1, 1))
        self.clear()
        return ImagineOutput(
            observations={
                k: torch.stack([o[k] for o in all_observations], dim=1)
                for k in self._ordered_modalities
            },  # (B, T, C, H, W)
            actions=actions[:, seq_len - effective_horizon - 1 : seq_len - 1],  # (B, T)
            actions_distributions=self._concat_distributions(
                all_actions_dists
            ),  # (B, T, #actions)
            values_info=self._concat_values(all_values_info),  # (B, T)
            q_values_info=None,  # self._concat_values(all_q_info),
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=ends[:, seq_len - effective_horizon - 1 : seq_len - 1],  # (B, T)
        )


class ActorCriticLS2(ActorCriticLS, ABC):

    def __init__(
        self,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        self.num_value_categories = num_value_bins + 1
        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

    def _build_critic_head(self):
        return HLGaussCategoricalRegressionHead(
            self.lstm_dim,
            self.num_value_categories,
            sym_log_normalize=True,
            device=self.device,
        )

    def _compute_critic_output(self, critic_latent) -> CategoricalCriticOutput:
        means_values = self.critic_v_head(critic_latent)

        return CategoricalCriticOutput(value_logits=critic_latent, values=means_values)

    def _compute_critic_loss(
        self, values_info: CategoricalValuesInfo, targets
    ) -> Tensor:
        loss = self.critic_v_head.compute_loss(
            values_info.values_logits[:, :-1], targets
        )
        return loss

    def _concat_values(self, values_info_list: list[CategoricalValuesInfo]):
        return CategoricalValuesInfo(
            values_logits=torch.stack(
                [vi.values_logits for vi in values_info_list], dim=1
            ),
            values=torch.stack([vi.values for vi in values_info_list], dim=1),
        )


class DiscreteActorCriticLS(ActorCriticLS2):

    def __init__(
        self,
        act_vocab_size,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        unknown_action=False,
        **kwargs,
    ) -> None:
        self.num_actions = act_vocab_size 

        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

        act_dim=self.num_actions if unknown_action else self.num_actions+1
        self.actor_actions_embeddings = nn.Embedding(
            act_dim, self.embed_dim, device=self.device
        )
        self.critic_actions_embeddings = (
            nn.Embedding(act_dim, self.embed_dim, device=self.device)
            if separate_networks
            else None
        )

    def _build_actor_head(self) -> nn.Module:
        return nn.Linear(self.lstm_dim, self.num_actions, device=self.device)

    def embed_action(self, action) -> tuple[Tensor, Tensor]:
        actor_action_emb = self.actor_actions_embeddings(action.long())
        critic_action_emb = (
            self.critic_actions_embeddings(action.long())
            if self.separate_networks
            else None
        )

        return actor_action_emb, critic_action_emb

    def _compute_actor_output(self, actor_latent) -> DiscreteActorOutput:
        logits_actions = self.actor_linear(actor_latent)
        if logits_actions.dim() == 2:
            logits_actions = logits_actions.unsqueeze(1)

        return DiscreteActorOutput(logits_actions=logits_actions)

    def _concat_distributions(self, all_actions_dists) -> CategoricalDistribution:
        return CategoricalDistribution(
            logits=torch.cat([d.logits for d in all_actions_dists], dim=1)
        )


class MultiDiscreteActorCriticLS(ActorCriticLS2):

    def __init__(
        self,
        actions_nvec,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        self.actions_nvec = actions_nvec

        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

        self.actor_actions_embeddings = MultiDiscreteEmbedding(
            actions_nvec, self.embed_dim, device=self.device
        )
        self.critic_actions_embeddings = (
            MultiDiscreteEmbedding(actions_nvec, self.embed_dim, device=self.device)
            if separate_networks
            else None
        )

    def _build_actor_head(self) -> nn.Module:
        return nn.Linear(self.lstm_dim, sum(self.actions_nvec), device=self.device)

    def embed_action(self, action) -> tuple[Tensor, Tensor]:
        actor_action_emb = self.actor_actions_embeddings(action.long()).mean(dim=-2)
        critic_action_emb = (
            self.critic_actions_embeddings(action.long()).mean(dim=-2)
            if self.separate_networks
            else None
        )

        return actor_action_emb, critic_action_emb

    def _compute_actor_output(self, actor_latent) -> MultiDiscreteActorOutput:
        logits_actions = self.actor_linear(actor_latent)
        if logits_actions.dim() == 2:
            logits_actions = logits_actions.unsqueeze(1)

        return MultiDiscreteActorOutput(
            logits_actions=logits_actions, nvec=self.actions_nvec
        )

    def _concat_distributions(self, all_actions_dists) -> MultiCategoricalDistribution:
        return MultiCategoricalDistribution(
            logits=torch.cat([d.logits for d in all_actions_dists], dim=1),
            nvec=all_actions_dists[0].nvec,
        )


class ContinuousActorCriticLS(ActorCriticLS2):
    def __init__(
        self,
        action_dim: int,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )
        self.actor_action_projection = nn.Linear(
            action_dim, self.embed_dim, device=device
        )
        self.critic_action_projection = (
            nn.Linear(action_dim, self.embed_dim, device=device)
            if separate_networks
            else None
        )

    def _build_actor_head(self) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(
                self.lstm_dim, self.action_dim * 2, device=self.device
            )  # mean, std
        )
        model[-1].bias.data = torch.cat(
            [
                torch.zeros(self.action_dim, device=self.device),
                torch.ones(self.action_dim, device=self.device) * 2,
            ]
        )
        return model

    def embed_action(self, action: Tensor) -> tuple[Tensor, Tensor]:
        critic_emb = (
            self.critic_action_projection(action) if self.separate_networks else None
        )
        return self.actor_action_projection(action), critic_emb

    def _compute_actor_output(self, actor_latent) -> ContinuousActorOutput:
        actor_outs = self.actor_linear(actor_latent)
        if actor_latent.dim() == 2:
            actor_outs = actor_outs.unsqueeze(1)
        actions_means, actions_log_stds = torch.split(
            actor_outs, self.action_dim, dim=-1
        )
        # actions_means = torch.clamp(actions_means, -1, 1)
        min_std, max_std = (0.05, 1)
        actions_stds = min_std + torch.sigmoid(actions_log_stds) * (max_std - min_std)
        # actions_stds = torch.ones_like(actions_log_stds)*0.1

        return ContinuousActorOutput(
            actions_means=actions_means, actions_stds=actions_stds
        )

    def _concat_distributions(self, all_actions_dists) -> Normal:
        return SquashedDiagNormalDistribution(
            loc=torch.cat([d.loc for d in all_actions_dists], dim=1),
            scale=torch.cat([d.scale for d in all_actions_dists], dim=1),
        )


class DContinuousActorCriticLS(ContinuousActorCriticLS):

    def __init__(
        self,
        action_dim: int,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 128,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        n_action_quant_levels: int = 51,
        **kwargs,
    ):
        self.n_action_quant_levels = n_action_quant_levels
        super().__init__(
            action_dim=action_dim,
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

    def _build_actor_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.lstm_dim,
                self.action_dim * self.n_action_quant_levels,
                device=self.device,
            )
        )

    def _compute_actor_output(self, actor_latent) -> QuantizedContinuousActorOutput:
        actions_logits = rearrange(
            self.actor_linear(actor_latent),
            "... (m n) -> ... m n",
            m=self.action_dim,
            n=self.n_action_quant_levels,
        )
        if actor_latent.dim() == 2:
            actions_logits = actions_logits.unsqueeze(1)

        return QuantizedContinuousActorOutput(logits_actions=actions_logits)

    def _concat_distributions(
        self, all_actions_dists
    ) -> QuantizedContinuousDistribution:
        return QuantizedContinuousDistribution(
            logits=torch.cat([d.logits for d in all_actions_dists], dim=1)
        )
