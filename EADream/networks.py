import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools

# torch.autograd.set_detect_anomaly(True)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=64):
        """实现位置编码"""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """输入x的形状为[batch_size, seq_len,  hidden_size]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        """实现Scaled Dot-Product Attention"""
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v的形状为[batch_size, num_heads, seq_len, hidden_size / num_heads]
        mask的形状为[batch_size, 1, seq_len, seq_len]
        output:
            - context: 输出值
            - attention: 计算得到的注意力矩阵
        """
        context, attention = None, None
        # TODO
        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            # pdb.set_trace()
            # print(mask.shape, scores.shape)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.softmax(scores)
        # pdb.set_trace()
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        """实现Multi-Head Attention"""
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v的形状为[batch_size, seq_len, hidden_size]
        mask的形状为[batch_size, 1, seq_len, seq_len]
        """
        residual = q
        batch_size = q.size(0)
        output, attention = None, None
        # TODO

        if q.equal(k):
            q = self.layer_norm(q)
            k = self.layer_norm(k)
            v = self.layer_norm(v)
        elif k.equal(v):
            q = self.layer_norm(q)

            # pdb.set_trace()
        # mask = mask.repeat(1, self.num_heads, 1, 1)

        d = self.hidden_size // self.num_heads
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d).transpose(1, 2)

        context, attention = self.scaled_dot_product_attention(q, k, v, mask=mask)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * d)
        )
        # pdb.set_trace()
        output = self.linear(context)  #  + residual

        output = self.dropout(output)
        output = output + residual
        return output, attention


class FeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        """实现FFN"""
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, filter_size)
        self.linear_2 = nn.Linear(filter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """输入x的形状为[batch_size, seq_len, hidden_size]"""
        residual = x
        x = self.layer_norm(x)
        output = self.linear_2(self.dropout(F.relu(self.linear_1(x))))
        output = self.dropout(output) + residual
        return output


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.attention_1 = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_2 = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = FeedForward(hidden_size, hidden_size * 2, dropout)

    def forward(self, x, encoder_output):
        x = self.attention_1(x, x, x, need_weights=False)[0]
        x = self.attention_2(x, encoder_output, encoder_output, need_weights=False)[0]
        x = self.feed_forward(x)
        return x


class StochDecoder(nn.Module):
    def __init__(self, discrete_dim, outdim, action_dim, config, act="SiLU", norm=True):
        super(StochDecoder, self).__init__()
        self.stochembedding = nn.Linear(discrete_dim, config["embed_dim"])
        self.actionembedding = nn.Linear(action_dim, config["embed_dim"])
        self.position_embedding = PositionalEncoding(
            config["embed_dim"], config["dropout"]
        )
        self.layer = nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        # nn.ModuleList(
        #     [
        #         DecoderLayer(hidden_size, config["num_heads"], config["dropout"])
        #         for _ in range(config["layers"])
        #     ]
        # )
        out_layers = []
        if norm:
            out_layers.append(nn.LayerNorm(config["embed_dim"], eps=1e-03))
        act = getattr(torch.nn, act)
        out_layers.append(act())
        out_layers.append(nn.Linear(config["embed_dim"], outdim, bias=False))
        self._outlayers = nn.Sequential(*out_layers)

    def forward(self, x, action):

        # ？RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operatio
        x = self.stochembedding(x)
        # x = self.position_embedding(x)
        action_embed = self.actionembedding(action.clone())
        action_embed2 = action_embed.reshape((action_embed.shape[0], 1, -1))
        x = self.layer(x, action_embed2, action_embed2, need_weights=False)[0]
        x = self._outlayers(x)
        x = x.reshape(x.shape[0], -1)
        return x


class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        mha=False,
        mha_config=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        self._mha = mha
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if mha and self._discrete:
            self._img_in_layers = StochDecoder(
                self._discrete,
                self._hidden // self._stoch,
                self._num_actions,
                mha_config,
            )
        else:
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            if norm:
                inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        print(f"obs_out_inpdim:{inp_dim}")
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            if self._mha:

                x = self._img_in_layers(prev_stoch, prev_action)
            else:
                shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
                # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
                prev_stoch = prev_stoch.reshape(shape)
                # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
                x = torch.cat([prev_stoch, prev_action], -1)
                # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
                x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            try:
                stoch = self.get_dist(stats).sample()
            except:
                print(
                    f"stats:{stats}; x:{x};prev_state:{prev_state},prev_act:{prev_action}"
                )
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        attention,
        attention_k=3,
        attention_r=2,
        stackframe=True
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward", "event")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)
        self.stackframe = stackframe
        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                attention,
                attention_k,
                attention_r,
                stackframe,
            )

            self.outdim += self._cnn.outdim
            if stackframe:
                self._3dcnn = ClipEncoder(
                    input_shape,
                    depth=cnn_depth * 2,
                    act="SiLU",
                    norm=True,
                    kernel_size=4,
                )
                self._dircnn = DirectionEncoder(
                    self._3dcnn.outputshape,
                    cnn_depth,
                    act,
                    norm,
                    kernel_size,
                    minres,
                    attention,
                    attention_k,
                    attention_r
                )

                self.outdim += self._dircnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs, batch=False):
        outputs = []
        if (
            self.cnn_shapes
        ):  # self.attention_weight*obs[k]*obs["event"]+(1-self.attention_weight)*
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            if self.stackframe:
                if batch == False:  # time=1
                    i, mot = self._3dcnn(inputs)
                else:
                    # (batch, time, h, w, ch) -> (batch, ch, time, h, w)
                    tpadinputs = inputs.permute(0, 4, 1, 2, 3)
                    tpadinputs = F.pad(tpadinputs, (0, 0, 0, 0, 2, 0), "replicate")
                    # (batch, ch, time, h, w) -> (batch, time, h, w, ch)
                    tpadinputs = tpadinputs.permute(0, 2, 3, 4, 1)
                    i, mot = self._3dcnn(tpadinputs)
                    outputs.append(self._cnn(i))
                    outputs.append(self._dircnn(mot))
            else:
                outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        if batch == False:
            outputs = outputs.squeeze(1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        device,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
        attention,
        attention_weight=0.5,
        dizzy=False,
        loss_scale=1.0,
        f_gamma=2,
        f_alpha=0.5,
        directionmot=False,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        self.attention_weight = (
            attention_weight  # takes effect only for image_dist=attetionmse
        )
        self.dizzy = dizzy
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        self.f_gamma = f_gamma
        self.f_alpha = f_alpha
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
                attention=attention,
                directionmot=directionmot,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist
        self.device = device
        self.directionmot = directionmot

    def forward(self, features):

        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        if self._image_dist == "lpips":
            return tools.LpipsDist(mean, self.device)
        if self._image_dist == "attentionmse":
            return tools.AttentionMSEDist(
                mean, attention_weight=self.attention_weight, dizzy=self.dizzy
            )
        if self._image_dist == "bce":
            return tools.BCEDist(mean)
        if self._image_dist == "focal":
            return tools.FocalBinaryDist(mean, gamma=self.f_gamma, alpha=self.f_alpha)
        raise NotImplementedError(self._image_dist)


class Harmonizer(nn.Module):
    def __init__(self, device, regularize=True):
        super(Harmonizer, self).__init__()
        self.device = device
        self.regularize = regularize
        self.harmony_s = nn.Parameter(torch.tensor([0.0], device=self.device))

    def forward(self, x):
        if self.regularize:
            return x / (torch.exp(self.harmony_s)) + torch.log(
                torch.exp(self.harmony_s) + 1
            )
        else:
            return x / (torch.exp(self.harmony_s))

    def get_harmony(self):
        return torch.exp(self.harmony_s).item()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return x * out


class ClipEncoder(nn.Module):
    def __init__(self, input_shape, depth=32, act="SiLU", norm=True, kernel_size=4):
        super(ClipEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape

        in_dim = input_ch
        out_dim = depth
        self.outdim = depth
        self.outputshape = (h // 2, w // 2, depth // 2)
        layers = []
        pad_h = calc_same_pad(i=h, k=kernel_size, s=2, d=1)
        pad_w = calc_same_pad(i=w, k=kernel_size, s=2, d=1)
        layers.append(
            nn.ZeroPad3d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, 0, 0)
            )
        )
        layers.append(
            nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(3, kernel_size, kernel_size),
                stride=(1, 2, 2),
            )
        )
        if norm:
            layers.append(Img3DChLayerNorm(out_dim))
        layers.append(act())
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        x = obs.permute(0, 4, 1, 2, 3)
        x = self.layers(x)
        # (batch, ch, time, h, w) -> (batch, time, h, w, ch)
        x = x.permute(0, 2, 3, 4, 1)
        return (x[..., : self.outdim // 2], x[..., self.outdim // 2 :])


# class FusionLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_size):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = Conv2dSamePad(
#             in_channels=in_dim,
#             out_channels=out_dim,
#             kernel_size=kernel_size,
#             stride=2,
#             bias=False,
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, flow):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv1(out)
#         out = self.sigmoid(out)
#         return x * out


# class FusionEncoder(nn.Module):
#     def __init__(
#         self,
#         input_shape,
#         depth=32,
#         act="SiLU",
#         norm=True,
#         kernel_size=4,
#         minres=4,
#         attention=False,
#         attention_k=3,
#         attention_r=2,
#         directions=8,
#         device="cuda:0",
#     ):
#         super(FusionEncoder, self).__init__()
#         act = getattr(torch.nn, act)
#         h, w, input_ch = input_shape

#         in_dim = input_ch
#         out_dim = depth
#         self.outdim = depth
#         self.outputshape = (h // 2, w // 2, depth // 2)
#         self.imagelayers = nn.ModuleList()
#         stages = int(np.log2(h) - np.log2(minres))
#         if attention:
#             self.imagelayers.append(ChannelAttention(in_dim, ratio=1))
#             self.imagelayers.append(SpatialAttention(kernel_size=attention_k))
#         self.imagelayers.append(
#             Conv2dSamePad(
#                 in_channels=in_dim,
#                 out_channels=out_dim,
#                 kernel_size=kernel_size,
#                 stride=2,
#                 bias=False,
#             )
#         )
#         if norm:
#             self.imagelayers.append(ImgChLayerNorm(out_dim))
#         self.imagelayers.append(act())
#         out_dim = depth

#         layers = []
#         pad_h = calc_same_pad(i=h, k=kernel_size, s=2, d=1)
#         pad_w = calc_same_pad(i=w, k=kernel_size, s=2, d=1)
#         layers.append(
#             nn.ZeroPad3d(
#                 (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, 0, 0)
#             )
#         )
#         layers.append(
#             nn.Conv3d(
#                 input_ch,
#                 directions,
#                 kernel_size=(2, kernel_size, kernel_size),
#                 stride=(1, 2, 2),
#             )
#         )
#         if norm:
#             layers.append(Img3DChLayerNorm(out_dim))
#         layers.append(act())
#         self.flowlayer1 = nn.Sequential(*layers)
#         self.flowlayer1.apply(tools.weight_init)
#         if attention:
#             self.layers.append(ChannelAttention(in_dim, ratio=attention_r))
#             self.layers.append(SpatialAttention(kernel_size=attention_k))
#         layers = []
#         pad_h = calc_same_pad(i=h, k=kernel_size, s=2, d=1)
#         pad_w = calc_same_pad(i=w, k=kernel_size, s=2, d=1)
#         layers.append(
#             nn.ZeroPad3d(
#                 (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, 0, 0)
#             )
#         )
#         layers.append(
#             nn.Conv3d(
#                 in_dim,
#                 out_dim,
#                 kernel_size=(2, kernel_size, kernel_size),
#                 stride=(1, 2, 2),
#             )
#         )

#         if norm:
#             layers.append(Img3DChLayerNorm(directions))
#         layers.append(act())
#         self.flowlayer1 = nn.Sequential(*layers)
#         self.flowlayer1.apply(tools.weight_init)

#         h, w = h // 2, w // 2
#         in_dim = out_dim
#         out_dim *= 2

#         x = torch.linspace(-1, 1, w)
#         y = torch.linspace(-1, 1, h)
#         grid_x, grid_y = torch.meshgrid(x, y)
#         grid = torch.stack([grid_x, grid_y], dim=0).to(device)
#         self.grid = grid.permute(0, 2, 1)  # (2, H, W)
#         self.directions = torch.stack(
#             (
#                 torch.cos(torch.arange(directions) / directions * 2 * math.pi),
#                 torch.sin(torch.arange(directions) / directions * 2 * math.pi),
#             ),
#             dim=1,
#         ).to(device)

#     def forward(self, obs):
#         obs -= 0.5
#         image1 = obs[:, 2:].clone()
#         # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
#         image1 = image1.reshape((-1,) + tuple(image1.shape[-3:]))
#         # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
#         image1 = image1.permute(0, 3, 1, 2)
#         image1 = self.imagelayers(image1)
#         # (batch, time, h, w, ch)->(batch, ch, time, h, w)
#         flowx = obs.permute(0, 4, 1, 2, 3)
#         flowx = self.flowlayer1(flowx)
#         image1 = F.grid_sample(
#             x,
#             grid=dirs[(i - first_index + 1) // self.module_per_layer - 1].reshape(
#                 (-1,) + tuple(x.shape[-2:]) + (2,)
#             ),
#             mode="bilinear",
#         )
#         # (batch, time, h, w, ch) -> (batch, time, ch, h, w)
#         x = obs.permute(0, 4, 1, 2, 3)
#         return (x[..., : self.outdim // 2], x[..., self.outdim // 2 :])


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        attention=False,
        attention_k=3,
        attention_r=2,
        stackframe=False,
        device="cuda:0",
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        self.layers = nn.ModuleList()
        self.attention = attention
        self.stackframe = stackframe
        if stackframe:
            in_dim = depth
        self.device = device
        if attention:
            self.layers.append(ChannelAttention(in_dim, ratio=1))
            self.layers.append(SpatialAttention(kernel_size=attention_k))
        for i in range(stages):
            if stackframe == False or i != 0:
                self.layers.append(
                    Conv2dSamePad(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=kernel_size,
                        stride=2,
                        bias=False,
                    )
                )
                if norm:
                    self.layers.append(ImgChLayerNorm(out_dim))
                self.layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2
        self.module_per_layer = 3 if norm else 2
        if attention:
            self.layers.append(ChannelAttention(in_dim, ratio=attention_r))
            self.layers.append(SpatialAttention(kernel_size=attention_k))
        self.outdim = out_dim // 2 * h * w
        self.hw = h * w
        self.ch = out_dim // 2
        print(f"encoder outdim: {self.outdim}")

        self.layers.apply(tools.weight_init)

    def forward(self, obs, dirs=None):
        if self.stackframe == False:
            obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        first_index = 2 if self.attention else 0
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if (
                dirs != None
                and (self.attention == False or i > 2 and i < len(self.layers) - 2)
                and (i - first_index + 1) % self.module_per_layer == 0
            ):
                w, h = x.shape[-2:]
                gx = torch.linspace(-1, 1, w)
                gy = torch.linspace(-1, 1, h)
                grid_x, grid_y = torch.meshgrid(gx, gy)
                grid = torch.stack([grid_x, grid_y], dim=0).to(x.device)
                grid = grid.permute(1, 2, 0)  # ( H, W,2)
                x = F.grid_sample(
                    x,
                    grid=dirs[
                        (i - first_index + 1) // self.module_per_layer - 1
                    ].reshape((-1,) + tuple(x.shape[-2:]) + (2,))
                    + grid,
                    mode="bilinear",
                )
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class DirectionEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        attention=False,
        attention_k=3,
        attention_r=2,
        directionnum=16,
        device="cuda:0",
    ):
        super(DirectionEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = directionnum
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(stages):
            pad_h = calc_same_pad(i=h, k=kernel_size, s=2, d=1)
            pad_w = calc_same_pad(i=w, k=kernel_size, s=2, d=1)
            self.layers.append(
                nn.ZeroPad3d(
                    (
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                        0,
                        0,
                    )
                )
            )
            self.layers.append(
                nn.Conv3d(
                    in_dim,
                    out_dim,
                    kernel_size=(1, kernel_size, kernel_size),
                    stride=(1, 2, 2),
                    padding_mode="replicate",
                )
            )
            h, w = h // 2, w // 2
            if norm:
                self.layers.append(Img3DChLayerNorm(out_dim))
            self.layers.append(act())
            in_dim = out_dim
            if i == 1:
                out_dim = directionnum
        self.outdim = 2 * h * w
        self.hw = h * w
        self.ch = directionnum
        self.module_per_layer = 4 if norm else 3
        self.directions = torch.stack(
            (
                torch.cos(torch.arange(self.ch) / self.ch * 2 * math.pi),
                torch.sin(torch.arange(self.ch) / self.ch * 2 * math.pi),
            ),
            dim=1,
        ).to(device)
        print(f"encoder outdim: {self.outdim}")
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch, ch, time, h, w)
        x = obs.permute(0, 4, 1, 2, 3)

        outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

        # (batch, ch, time, h, w) -> (batch, time, h, w, ch)

        x = F.softmax(x.permute(0, 2, 3, 4, 1), dim=-1) @ self.directions
        x = x.reshape(list(obs.shape[:-3]) + [-1])
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
        attention=True,
        attention_k=3,
        attention_r=2,
        directionmot=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            # if attention:
            #     layers.append(ChannelAttention(in_dim,ratio=1))
            #     layers.append(SpatialAttention(kernel_size=3))
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            if h == 16:
                [m.apply(tools.weight_init) for m in layers[:-1]]
                self.layer1 = nn.Sequential(*layers)
                layers = []
                if directionmot:
                    in_dim += 2
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            if h == 16 and directionmot:
                [m.apply(tools.weight_init) for m in layers[:-1]]
                self.dirmotconvlayer = nn.Sequential(*layers)
                layers = []
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2

        if attention:
            # layers.append(ChannelAttention(in_dim,ratio=attention_r))
            layers.append(SpatialAttention(kernel_size=attention_k))

        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layer2 = nn.Sequential(*layers)
        self.directionmot = directionmot

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        if isinstance(features, tuple):
            features, dirmot = features
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layer1(x)
        if self.directionmot:
            dirmot = dirmot.reshape(-1, 2, 16, 16)
            x = torch.cat((x, dirmot), dim=1)
            x = self.dirmotconvlayer(x)
        x = self.layer2(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        # print(f"x.shape{x.shape}")
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


def calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


class Conv2dSamePad(torch.nn.Conv2d):
    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Img3DChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(Img3DChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x
