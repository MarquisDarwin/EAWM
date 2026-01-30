import copy
import torch
from torch import nn

import networks
import tools
import numpy as np
import torch.nn.functional as F

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(
            shapes, stackframe=config.stackframe, **config.encoder
        )
        self.embed_size = self.encoder.outdim
        print(f"encoder_embedsize:{self.embed_size}")
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.mha,
            config.mha_layer,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.harmony:
            self.harmonylist = ["reward", "rep"]

            if self.encoder.cnn_shapes:
                # if config.event_pred:
                #     self.harmonylist.append("event")
                for k in self.encoder.cnn_shapes.keys():
                    self.harmonylist.append(k)
            if self.encoder.mlp_shapes:
                for k in self.encoder.mlp_shapes.keys():
                    self.harmonylist.append(k)
            self.headscoeff = nn.ModuleDict()
            for h in self.harmonylist:
                self.headscoeff[h] = networks.Harmonizer(config.device)
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        if config.dyn_discrete:
            event_feat_size = (
                config.dyn_stoch * config.dyn_discrete * 2 + config.dyn_deter
            )
        else:
            event_feat_size = config.dyn_stoch * 2 + config.dyn_deter
        if config.event_pred:
            if config.prev_cur_input_event:
                event_feat_size = feat_size * 2
            self.heads["event"] = networks.MultiDecoder(
                event_feat_size,
                {"event": tuple(list(shapes["image"])[0:2] + [1])},
                device=self._config.device,
                **config.eventpred_head,
            )  # attention_weight
        if config.mae_ratio < 1.0:
            decfeat_size = (
                self.encoder._cnn.hw - int(self.encoder._cnn.hw * config.mae_ratio)
            ) * self.encoder._cnn.ch
        else:
            decfeat_size = feat_size
        self.heads["decoder"] = networks.MultiDecoder(
            decfeat_size,
            shapes,
            device=self._config.device,
            dizzy=self._config.dizzy,
            **config.decoder,
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        if config.event_pred and "image" in self.encoder.cnn_shapes.keys():
            self._scales["eventpred"] = (config.eventpred_head["loss_scale"],)
        self._event_pred_ratio = config.event_pred_ratio

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data, batch=True)

                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                if self._config.harmony:
                    kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                        post,
                        prior,
                        kl_free,
                        self._config.dyn_scale,
                        self._config.rep_scale,
                    )
                    kl_loss = self.headscoeff["rep"](kl_loss)
                    dyn_loss = dyn_loss * self.headscoeff["rep"].get_harmony()
                    rep_loss = rep_loss * self.headscoeff["rep"].get_harmony()
                    dyn_scale = self.headscoeff["rep"].get_harmony()
                    rep_scale = self.headscoeff["rep"].get_harmony()
                else:
                    dyn_scale = self._config.dyn_scale
                    rep_scale = self._config.rep_scale
                    kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                        post, prior, kl_free, dyn_scale, rep_scale
                    )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    if name == "decoder":
                        if self._config.mae_ratio < 1.0:
                            x = embed
                            batch, time = x.shape[0], x.shape[1]
                            nc = self.encoder._cnn.hw
                            mc = nc - int(nc * self._config.mae_ratio)
                            # (batch, time, embed_dim) ->(batch, time, h * w, ch)
                            x = x.reshape((batch, time, nc, -1))
                            # (batch, time, h * w, ch) ->(batch * time, h * w, ch)
                            x = x.reshape((-1,) + tuple(x.shape[2:]))
                            shuffle_indices = torch.rand(
                                batch * time, nc, device=self._config.device
                            ).argsort()
                            # indices of unmask and masked patches
                            unmask_ind, mask_ind = (
                                shuffle_indices[:, :mc],
                                shuffle_indices[:, mc:],
                            )

                            # batchtime indices：(batchtime,1)
                            batch_ind = torch.arange(
                                batch * time, device=self._config.device
                            ).unsqueeze(-1)
                            # sample patches in terms of indices，divided into mask and unmasked

                            # (batch * time, h * w, ch) -> (batch * time, ch, h * w)
                            feat = x[batch_ind, unmask_ind].permute(0, 2, 1)
                            feat = feat.reshape([batch, time, -1])
                        else:
                            if self._config.predictfromprior:
                                feat = torch.cat(
                                    [
                                        self.dynamics.get_feat(post)[:, 0].unsqueeze(1),
                                        self.dynamics.get_feat(prior)[:, 1:],
                                    ],
                                    dim=1,
                                )
                            else:
                                feat = self.dynamics.get_feat(post)
                            if (
                                self._config.stackframe
                                and self._config.decoder["directionmot"]
                            ):
                                feat = (feat, embed[:, :, -512:])
                    elif name == "event":
                        deter = post["deter"]
                        prior_stoch = prior["stoch"]
                        post_stoch = post["stoch"]
                        if self._config.dyn_discrete:
                            pshape = list(prior_stoch.shape[:-2]) + [
                                self._config.dyn_discrete * self._config.dyn_stoch
                            ]
                            prior_stoch = prior_stoch.reshape(pshape)
                            post_stoch = post_stoch.reshape(pshape)
                        if self._config.prev_cur_input_event:
                            feat = torch.cat(
                                [
                                    post_stoch[:, :-1],
                                    prior_stoch[:, 1:],
                                    deter[:, :-1],
                                    deter[:, 1:],
                                ],
                                dim=-1,
                            )  # time_dim=63
                        else:
                            feat = torch.cat(
                                [prior_stoch, post_stoch, deter], dim=-1
                            )  # time_dim=64

                    else:
                        feat = self.dynamics.get_feat(post)
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    if name == "event":
                        if self._config.prev_cur_input_event:
                            eventtruth = data[name][:, 1:]
                        else:
                            eventtruth = data[name]
                        pixels = np.prod(eventtruth.shape[2:])
                        loss0 = -pred.log_prob(eventtruth)
                        event_num = torch.sum(
                            eventtruth, dim=list(range(2, 2 + len(eventtruth.shape[2:])))
                        )
                        if self._config.prev_cur_input_event == False:
                            loss0[:, 0] = 0
                        if self._config.dizzy:
                            loss = torch.where(
                                event_num < self._event_pred_ratio * pixels, loss0, 0
                            )
                    elif name == "image":
                        if self._config.decoder["image_dist"] == "attentionmse":
                            loss = -pred.log_prob(
                                data[name], data["event"], self._event_pred_ratio
                            )
                        else:
                            loss = -pred.log_prob(data[name])
                        if self._config.first_frame_pred == False:
                            loss[:, 0] = 0
                    elif name == "reward" or name == "cont":
                        loss = -pred.log_prob(data[name])
                    else:  # obs to be reconstructed or predicted
                        loss = -pred.log_prob(data[name])
                        if self._config.first_frame_pred == False:
                            loss[:, 0] = 0
                    # assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = torch.mean(loss)
                if self._config.harmony:
                    scaled = {}
                    for key, value in losses.items():
                        if key in self.headscoeff.keys():
                            scaled[key] = self.headscoeff[key](value)
                    scaled["cont"] = (
                        losses["cont"] * self._config.cont_head["loss_scale"]
                    )
                    if (
                        self._config.event_pred
                        and "image" in self.encoder.cnn_shapes.keys()
                    ):
                        scaled["event"] = (
                            losses["event"] * self._config.eventpred_head["loss_scale"]
                        )
                else:
                    scaled = {
                        key: value * self._scales.get(key, 1.0)
                        for key, value in losses.items()
                    }
                model_loss = sum(scaled.values()) + torch.mean(kl_loss)
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        if self._config.harmony:
            metrics.update(
                {
                    f"{name}_scale": scale.get_harmony()
                    for name, scale in self.headscoeff.items()
                }
            )
        else:
            metrics["dyn_scale"] = dyn_scale
            metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"])

        obs["image"] /= 255.0
        obs["event"] = (torch.Tensor(obs["event"]) / 255.0).unsqueeze(-1)

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def timepad(self, image):
        image = image.pe

    def video_pred(self, data):
        with torch.no_grad():
            data = self.preprocess(data)
            embed = self.encoder(data, batch=True)
            post, prior = self.dynamics.observe(
                embed[:6 ], data["action"][:6], data["is_first"][:6]
            )
            if self._config.mae_ratio < 1.0:
                x = embed[:6]
                batch, time = x.shape[0], x.shape[1]
                nc = self.encoder._cnn.hw
                mc = nc - int(nc * self._config.mae_ratio)
                # (batch, time, embed_dim) ->(batch, time, h * w, ch)
                x = x.reshape((batch, time, nc, -1))
                # (batch, time, h * w, ch) ->(batch * time, h * w, ch)
                x = x.reshape((-1,) + tuple(x.shape[2:]))
                shuffle_indices = torch.rand(
                    batch * time, nc, device=self._config.device
                ).argsort()
                # indices of unmask and masked patches
                unmask_ind, mask_ind = shuffle_indices[:, :mc], shuffle_indices[:, mc:]

                # batchtime indices：(batchtime,1)
                batch_ind = torch.arange(
                    batch * time, device=self._config.device
                ).unsqueeze(-1)
                # sample patches in terms of indices，divided into mask and unmasked

                # (batch * time, h * w, ch) -> (batch * time, ch, h * w)
                feat = x[batch_ind, unmask_ind].permute(0, 2, 1)
                feat = feat.reshape((batch, time, -1))
                recon = self.heads["decoder"](feat)["image"].mode()
                model = recon
            else:
                if self._config.predictfromprior:
                    feat = torch.cat(
                        [
                            self.dynamics.get_feat(post)[:, 0].unsqueeze(1),
                            self.dynamics.get_feat(prior)[:, 1:5],
                        ],
                        dim=1,
                    )
                else:
                    feat = self.dynamics.get_feat(post)[:5]
                if self._config.stackframe and self._config.decoder["directionmot"]:
                    recon = self.heads["decoder"]((feat, embed[:6, :5, -512:]))[
                        "image"
                    ].mode()[:6]
                else:
                    recon = self.heads["decoder"](feat)["image"].mode()[:6]
                    # mask reconstruction
                    if self._config.event_pred:
                        prior_stoch = prior["stoch"]
                        post_stoch = post["stoch"]
                        deter = post["deter"]

                        if self._config.dyn_discrete:
                            pshape = list(prior_stoch.shape[:-2]) + [
                                self._config.dyn_discrete * self._config.dyn_stoch
                            ]
                            prior_stoch = prior_stoch.reshape(pshape)
                            post_stoch = post_stoch.reshape(pshape)
                        if self._config.prev_cur_input_event:
                            feat = torch.cat(
                                [
                                    post_stoch[:, :-1],
                                    prior_stoch[:, 1:],
                                    deter[:, :-1],
                                    deter[:, 1:],
                                ],
                                dim=-1,
                            )  # time_dim=4
                        else:
                            feat = torch.cat(
                                [prior_stoch, post_stoch, deter], dim=-1
                            )  # time_dim=5
                        remask = self.heads["event"](feat)["event"].mode()[:6, 1:] > 0.5
                # reward_post = self.heads["reward"](self.dynamics.get_feat(post)).mode()[:6]

                init = {k: v[:, -1] for k, v in post.items()}
                prior2 = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
                feat2 = self.dynamics.get_feat(prior2)

                if self._config.stackframe:
                    openl = self.heads["decoder"]((feat2, embed[:6, 5:, -512:]))[
                        "image"
                    ].mode()[:6]
                else:
                    openl = self.heads["decoder"](feat2)["image"].mode()[:6]

                initmask = torch.zeros_like(data["event"][:6, 0]).unsqueeze(1)
                if self._config.event_pred:
                # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior2)).mode()
                # observed image is given until 5 steps
                    predicted_mask = torch.cat([initmask, remask], 1).repeat(
                        1, 1, 1, 1, 3
                    )
                # (B,T,H,W,3)
                model = torch.cat([recon[:, :5], openl], 1)
            truth = data["image"][:6]
            error = (model - truth + 1.0) / 2.0
            if self._config.event_pred:
                return torch.cat([truth, model, error, predicted_mask], 2)
            else:
                return torch.cat([truth, model, error], 2)

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
