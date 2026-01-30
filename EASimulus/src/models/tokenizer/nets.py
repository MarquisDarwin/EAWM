"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float
    interp_mode: str


class Encoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.ch_mult)
        temb_ch = 0  # timestep embedding #channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(config.in_channels,
                                       config.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.config.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        temb = None  # timestep embedding

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        temb_ch = 0
        self.num_resolutions = len(config.ch_mult)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(config.ch_mult)
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        print(f"Tokenizer : shape of latent is {config.z_channels, curr_res, curr_res}.")

        # z to block_in
        self.conv_in = torch.nn.Conv2d(config.z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(config.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True, interp_mode=config.interp_mode)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None  # timestep embedding

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(
        self, in_channels: int, with_conv: bool, interp_mode: str = "nearest", half_out_channels: bool = False
    ) -> None:
        super().__init__()
        self.interp_mode = interp_mode
        self.with_conv = with_conv
        if self.with_conv:
            out_channels = in_channels if not half_out_channels else in_channels // 2
            self.conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=self.interp_mode)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool, double_out_channels: bool = False) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            out_channels = in_channels * 2 if double_out_channels else in_channels
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = None, conv_shortcut: bool = False,
                 dropout: float, temb_channels: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)      # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)        # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


@dataclass
class SimpleEncoderConfig:
    in_channels: int
    out_channels: int
    out_channels_1st_layer: int
    input_resolution: tuple[int, int]
    num_downsample_steps: int
    norm_num_groups: int
    downsample_use_conv: bool

    @property
    def z_channels(self):
        return self.out_channels


class SimpleEncoder(nn.Module):
    def __init__(self, config: SimpleEncoderConfig):
        super().__init__()
        self.config = config
        latent_dim = config.out_channels_1st_layer

        in_conv = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=latent_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.n_layers = config.num_downsample_steps
        blocks = nn.Sequential(*[
            nn.Sequential(
                Normalize(
                    num_groups=config.norm_num_groups,
                    in_channels=latent_dim * (2 ** i)
                ),
                # nn.BatchNorm2d(latent_dim * (2 ** i)),
                nn.SiLU(),
                Downsample(latent_dim * (2 ** i), with_conv=config.downsample_use_conv, double_out_channels=True),
                # Normalize(
                #     num_groups=config.norm_num_groups,
                #     in_channels=latent_dim * (2 ** (i+1))
                # ),
                # nn.SiLU(),
                # nn.Conv2d(
                #     in_channels=latent_dim * (2 ** (i+1)),
                #     out_channels=latent_dim * (2 ** (i+1)),
                #     kernel_size=3,
                #     stride=1,
                #     padding=1
                # )
            )
            for i in range(self.n_layers)
        ])

        out_layer_in_channels = latent_dim * (2 ** config.num_downsample_steps)
        out_conv = nn.Sequential(
            Normalize(in_channels=out_layer_in_channels, num_groups=config.norm_num_groups),
            # nn.BatchNorm2d(out_layer_in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_layer_in_channels,
                out_channels=config.out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.cnn = nn.Sequential(
            in_conv,
            blocks,
            out_conv,
        )

    def forward(self, x: torch.Tensor):
        return self.cnn(x)


class SimpleDecoder(nn.Module):
    def __init__(self, config: SimpleEncoderConfig):
        super().__init__()
        self.config = config
        latent_dim = config.out_channels_1st_layer
        code_dim = latent_dim * (2 ** config.num_downsample_steps)

        in_conv = nn.Sequential(
            nn.BatchNorm2d(config.out_channels),
            nn.Conv2d(
                in_channels=config.out_channels,
                out_channels=code_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.n_layers = config.num_downsample_steps
        blocks = nn.Sequential(*[
                nn.Sequential(
                    Normalize(max(code_dim // (2 ** i), latent_dim), num_groups=config.norm_num_groups),
                    # nn.BatchNorm2d(max(code_dim // (2 ** i), latent_dim)),
                    nn.SiLU(),
                    Upsample(
                        max(code_dim // (2 ** i), latent_dim),
                        with_conv=config.downsample_use_conv,
                        interp_mode="nearest-exact",
                        half_out_channels=(code_dim // (2 ** i) > latent_dim)
                    ),
                    # Normalize(max(code_dim // (2 ** (i+1)), latent_dim)),
                    # nn.SiLU(),
                    # nn.Conv2d(
                    #     in_channels=max(code_dim // (2 ** (i+1)), latent_dim),
                    #     out_channels=max(code_dim // (2 ** (i+1)), latent_dim),
                    #     kernel_size=3,
                    #     stride=1,
                    #     padding=1,
                    # )
                )
                for i in range(self.n_layers)
            ]
        )

        out_block = nn.Sequential(
            Normalize(
                in_channels=latent_dim, num_groups=config.norm_num_groups
            ),
            # nn.BatchNorm2d(latent_dim),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=config.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.cnn = nn.Sequential(
            in_conv,
            blocks,
            out_block,
        )

    def forward(self, x: torch.Tensor):
        return self.cnn(x)


