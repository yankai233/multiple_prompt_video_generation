import torch
import torch.nn as nn  # 网络
import torch.nn.functional as F  # 激活函数
import torchvision.datasets  # 样本
from torch.utils import *  # 图案管理
from torch.utils import data  # 预处理数据集
from torchvision import transforms  # 工具
import torch.optim as optim
from tqdm import tqdm  # 进度条
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Any, Dict, Optional, Tuple, Union
import logging
import colorlog

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel, CogVideoXPatchEmbed, CogVideoXBlock
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import get_3d_sincos_pos_embed, Timesteps, TimestepEmbedding
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0


class CogVideoXLayerNormZero_TI2V(CogVideoXLayerNormZero):
    def __init__(
            self,
            conditioning_dim: int,
            embedding_dim: int,
            elementwise_affine: bool = True,
            eps: float = 1e-5,
            bias: bool = True,
    ) -> None:
        super().__init__(conditioning_dim, embedding_dim, elementwise_affine, eps, bias)
        self.linear2 = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_hidden_states_img, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        enc_shift_img, enc_scale_img, enc_gate_img = self.linear2(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        encoder_hidden_states_img = self.norm(encoder_hidden_states_img) * (1 + enc_scale_img)[:, None, :] + enc_shift_img[:, None, :]

        return hidden_states, encoder_hidden_states, encoder_hidden_states_img, gate[:, None, :], enc_gate[:, None, :], enc_gate_img[:, None, :]


class CogVideoXBlock_TI2V(CogVideoXBlock):
    def __init__(
            self,
            dim: int,  # dim
            num_attention_heads: int,  # num_heads
            attention_head_dim: int,  # attn_head_dim
            time_embed_dim: int,  # time_embed_dim

            dropout: float = 0.0,  # dropout
            activation_fn: str = "gelu-approximate",  # 激活函数
            attention_bias: bool = False,  # attn b
            qk_norm: bool = True,  # qk norm
            norm_elementwise_affine: bool = True,  # 归一化
            norm_eps: float = 1e-5,  # 归一化
            final_dropout: bool = True,  # dropout
            ff_inner_dim: Optional[int] = None,  # ff_inner_dim: 4
            ff_bias: bool = True,  # ff b
            attention_out_bias: bool = True,  # attn out b
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, time_embed_dim, dropout, activation_fn,
                         attention_bias, qk_norm, norm_elementwise_affine, norm_eps, final_dropout, ff_inner_dim,
                         ff_bias, attention_out_bias)

        # 1. Self Attention
        # 自适应层归一化

        self.norm1 = CogVideoXLayerNormZero_TI2V(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # 2. Feed Forward
        # 自适应层归一化
        self.norm2 = CogVideoXLayerNormZero_TI2V(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

    def forward(
            self,
            hidden_states: torch.Tensor,  # [b,num_patches,embed_dim]
            encoder_hidden_states: torch.Tensor,  # [b,text_seq_len,embed_dim]
            encoder_hidden_states_image: torch.Tensor,  # [b,img_seq_len,embed_dim]

            temb: torch.Tensor,  # [b,time_dim]
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  #

    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        image_seq_length = encoder_hidden_states_image.size(1)
        # norm & modulate
        # gate[:, None, :]         [b,1,embedding_dim]
        norm_hidden_states, norm_encoder_hidden_states, norm_encoder_hidden_states_image, gate_msa, enc_gate_msa, enc_img_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, encoder_hidden_states_image, temb
        )

        # attention: 拼接后做自注意力
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=torch.cat([norm_encoder_hidden_states, norm_encoder_hidden_states_image], dim=1),
            image_rotary_emb=image_rotary_emb,
        )
        attn_encoder_hidden_states, attn_encoder_hidden_states_img = attn_encoder_hidden_states[:, :text_seq_length, :], attn_encoder_hidden_states[:, text_seq_length:, :]
        # 门控
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        encoder_hidden_states_image = encoder_hidden_states_image + enc_img_gate_msa * attn_encoder_hidden_states_img

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, norm_encoder_hidden_states_image, gate_ff, enc_gate_ff, enc_img_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, encoder_hidden_states_image, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_encoder_hidden_states_image, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        # 门控
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length + image_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        encoder_hidden_states_image = encoder_hidden_states_image + enc_img_gate_ff * ff_output[:,
                                                                                      text_seq_length: text_seq_length + image_seq_length]
        return hidden_states, encoder_hidden_states, encoder_hidden_states_image


if __name__ == '__main__':
    cogvideo_ti2v = CogVideoXBlock_TI2V(1920, 30, 64, 512)
    hidden_states = torch.randn(3, 1000, 1920)
    encoder_hidden_states = torch.randn(3, 226, 1920)
    encoder_hidden_states_image = torch.randn(3, 226, 1920)
    temb = torch.randn(3, 512)
    # 只会对hidden_states使用
    image_rotary_emb = [torch.randn(1000,  64), torch.randn(1000,  64)]
    hidden_states, encoder_hidden_states, encoder_hidden_states_image = cogvideo_ti2v(hidden_states, encoder_hidden_states, encoder_hidden_states_image, temb, image_rotary_emb=image_rotary_emb)
    print(hidden_states.shape)
    print(encoder_hidden_states.shape)
    print(encoder_hidden_states_image.shape)