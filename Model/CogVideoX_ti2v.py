"""
CogVideoX_ti2v:
    输入: prompt, 锚点帧视觉特征  embedding([b,l,512])
    输出: video       [b,f,c,h,w]

"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../../'))

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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Any, Dict, Optional, Tuple, Union
import logging
import colorlog

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel, CogVideoXPatchEmbed, CogVideoXBlock
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import get_3d_sincos_pos_embed, Timesteps, TimestepEmbedding
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from accelerate.logging import get_logger
from Mutiple_prompt_mutiple_scene.Script.finetune.constants import LOG_NAME, LOG_LEVEL

from Mutiple_prompt_mutiple_scene.Model.blocks import *
from Mutiple_prompt_mutiple_scene.Model.transformer_blocks import activation_fn

logger = get_logger(LOG_NAME, LOG_LEVEL)


class Decoupled_Module(nn.Module):
    """解耦网络: P(vision_feature|cross_attn_Map, text_embed), 将视觉语义和文本语义解耦"""
    def __init__(self,
                 text_squ_len=226,  # 输出的token数量(tokenizer的输出token数量)
                 proj_dim=4096,     # hidden_dim
                 activation='gelu'  # 激活函数
                 ):
        super().__init__()
        self.text_squ_len = text_squ_len
        self.proj_dim = proj_dim
        self.activation = activation_fn[activation]
        # Layer Normalization
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.output_norm = nn.LayerNorm(proj_dim)

        # Linear layers for projection
        self.text_proj = nn.Linear(proj_dim, proj_dim)
        self.attn_proj = nn.Linear(proj_dim, proj_dim)

        # Cross Attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=8)

        # Final projection layer
        self.final_proj = nn.Linear(proj_dim, proj_dim)

    def forward(self, current_text_embed, cross_attn_map, sum_feature=False):
        """
        Args:
            current_text_embed: (batch_size, text_squ_len, proj_dim) 文本CLIP embedding
            cross_attn_map: (batch_size, text_squ_len, proj_dim) 交叉注意力图
        Returns:
            vision_feature: (batch_size, proj_dim) 解耦后的图像特征
        """
        assert current_text_embed.shape == cross_attn_map.shape, f'current_text_embed.shape={current_text_embed.shape}, cross_attn_map.shape={cross_attn_map.shape}'

        # Normalize inputs
        text_proj = self.norm1(self.activation(self.text_proj(current_text_embed)))
        attn_proj = self.norm2(self.activation(self.attn_proj(cross_attn_map)))

        # Cross Attention
        # Query: text embedding, Key/Value: attention map
        attn_output, _ = self.cross_attn(query=text_proj, key=attn_proj, value=attn_proj)

        weighted_attn = attn_output * cross_attn_map
        if sum_feature:
            weighted_attn = weighted_attn.sum(dim=1)
        # Final projection
        vision_feature = self.output_norm(self.activation(self.final_proj(weighted_attn)))

        return vision_feature


class CogVideoXPatchEmbed_TI2V(CogVideoXPatchEmbed):
    """
    1. 将z_t分块
    2. 线性映射video_embedding,text_embedding,image_embedding
    3. 加位置编码
    """
    def __init__(
            self,
            patch_size: int = 2,  # patch_size
            patch_size_t: Optional[int] = None,  # patch_size_t
            in_channels: int = 16,  # c
            embed_dim: int = 1920,  # embed_dim
            text_embed_dim: int = 4096,  # text_embed_dim
            bias: bool = True,  # b
            sample_width: int = 90,  # latent w
            sample_height: int = 60,  # latent h
            sample_frames: int = 49,  # pixel f
            temporal_compression_ratio: int = 4,  # 时间压缩率
            max_text_seq_length: int = 226,  # max_squ_len
            spatial_interpolation_scale: float = 1.875,  # 空间插值
            temporal_interpolation_scale: float = 1.0,  # 时间插值
            use_positional_embeddings: bool = True,  # 使用位置编码
            use_learned_positional_embeddings: bool = True,  # 可学习的位编码

            image_embedding_dim: Optional[int] = None,
            max_image_seq_length: Optional[int] = None,

    ):
        self.image_embedding_dim = image_embedding_dim or text_embed_dim
        self.max_image_seq_length = max_image_seq_length or max_text_seq_length

        super().__init__(patch_size, patch_size_t, in_channels, embed_dim, text_embed_dim, bias, sample_width, sample_height, sample_frames, temporal_compression_ratio, max_text_seq_length, spatial_interpolation_scale, temporal_interpolation_scale, use_positional_embeddings, use_learned_positional_embeddings)
        self.image_proj = nn.Linear(image_embedding_dim, embed_dim)

        # 可学习的位置编码
        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(
            self, sample_height: int, sample_width: int, sample_frames: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Return:
            joint_pos_embedding: [1, text_seq_len + img_seq_len + num_patches, embed_dim]
        """
        # TI2V
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )
        # [patch_w * patch_h * patch_t, embed_dim]
        pos_embedding = pos_embedding.flatten(0, 1)
        # [1, text_seq_len + num_patches, embed_dim]
        joint_pos_embedding = pos_embedding.new_zeros(
            1, self.max_text_seq_length + self.max_image_seq_length + num_patches, self.embed_dim, requires_grad=False
        )
        # [1, text_seq_len + num_patches, embed_dim], 将patches的位置编码复制到joint_pos_embedding中
        joint_pos_embedding.data[:, self.max_text_seq_length + self.max_image_seq_length:].copy_(pos_embedding)

        return joint_pos_embedding

    def forward(self, text_embeds: torch.Tensor, video_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        r"""
        1. 对text_embeds linear          text_embeds: [b, text_seq_len, embed_dim]
        2. 对image_embeds linear         image_embeds: [b, image_seq_len, embed_dim]
        3. 对latent分块 + Linear           video_embeds: [b, num_patches, embed_dim]
        4. 拼接                       embeds: [b, text_seq_len + image_seq_len + num_patches, embed_dim]
        5. 加入位置编码

        Returns:
            TI2V embed: [b, text_seq_len + image_seq_len + num_patches, embed_dim]
            or
            T2V embes: [b, text_seq_len + num_patches, embed_dim]
        """
        # [b, text_seq_len, embed_dim]
        text_embeds = self.text_proj(text_embeds)
        image_embeds = self.image_proj(image_embeds)

        batch_size, num_frames, channels, height, width = video_embeds.shape

        # 分块为序列 + linear:  image_embeds: [b, num_patches, embed_dim]
        if self.patch_size_t is None:
            # [b * f, c, h, w]
            video_embeds = video_embeds.reshape(-1, channels, height, width)
            # [b * f, embed_dim, h, w]
            video_embeds = self.proj(video_embeds)
            # [b, f, embed_dim , h , w]
            video_embeds = video_embeds.view(batch_size, num_frames, *video_embeds.shape[1:])
            # [b, f, h * w, embed_dim]
            video_embeds = video_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            # [b, f * h * w, embed_dim]
            video_embeds = video_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t
            # [b,f,c,h,w] -> [b,f,h,w,c]
            video_embeds = video_embeds.permute(0, 1, 3, 4, 2)
            # [b, f//p_t, p_t, h//p, p, w//p, p, c]
            video_embeds = video_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            # [b, f//p_t * h//p * w//p, p_t * p * p * c]
            video_embeds = video_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            # [b, f//p_t * h//p * w//p, p_t * p * p * c] -> [b, num_patches, embed_dim]
            video_embeds = self.proj(video_embeds)
        # [b, text_seq_len + num_patches, embed_dim]

        embeds = torch.cat(
                [text_embeds, image_embeds, video_embeds], dim=1
            ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        # 加位置编码
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):      # 创建新的位置编码
                pos_embedding = self._get_positional_embeddings(
                        height, width, pre_time_compression_frames, device=embeds.device
                    )

            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        return embeds


class CogVideoXTransformer3D_TI2V(CogVideoXTransformer3DModel):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 30,  # num_heads
            attention_head_dim: int = 64,  # head_dim

            in_channels: int = 16,  # in_channel
            out_channels: Optional[int] = 16,  # out_channel

            flip_sin_to_cos: bool = True,  # 时间编码
            freq_shift: int = 0,  # 时间编码
            time_embed_dim: int = 512,  # 时间编码dim
            ofs_embed_dim: Optional[int] = None,  # out dim in ofs used in CogVideoX1.5-I2V
            text_embed_dim: int = 4096,  # text embedding dim

            num_layers: int = 30,  # num_layers
            dropout: float = 0.0,  # dropout
            attention_bias: bool = True,  # attn b

            sample_width: int = 90,  # latent w
            sample_height: int = 60,  # latent h
            sample_frames: int = 49,  # pixel f
            patch_size: int = 2,  # patch_size
            patch_size_t: Optional[int] = None,  # patch_size_t
            temporal_compression_ratio: int = 4,  # 空间压缩率

            max_text_seq_length: int = 226,  # text embedding max_squ_len

            activation_fn: str = "gelu-approximate",  # 激活函数
            timestep_activation_fn: str = "silu",  # 激活函数
            norm_elementwise_affine: bool = True,  # 归一化
            norm_eps: float = 1e-5,  # 归一化
            spatial_interpolation_scale: float = 1.875,  # 空间插值
            temporal_interpolation_scale: float = 1.0,  # 时间插值
            use_rotary_positional_embeddings: bool = False,  # 使用RoPE位置编码
            use_learned_positional_embeddings: bool = False,  # 使用可学习的位置编码
            patch_bias: bool = True,  # patch b

            # 图像注入参数
            image_embedding_dim: int = 4096,
            max_image_seq_length: int = 226,

            # 解耦网络参数
            use_decoupled_module: bool = True,
            decoupled_activation='gelu',
            logger: Optional[logging.Logger] = None,
            **kwargs
    ):
        super().__init__(num_attention_heads, attention_head_dim, in_channels, out_channels, flip_sin_to_cos, freq_shift, time_embed_dim, ofs_embed_dim, text_embed_dim, num_layers, dropout, attention_bias, sample_width, sample_height, sample_frames, patch_size, patch_size_t, temporal_compression_ratio, max_text_seq_length, activation_fn, timestep_activation_fn, norm_elementwise_affine, norm_eps, spatial_interpolation_scale, temporal_interpolation_scale, use_rotary_positional_embeddings, use_learned_positional_embeddings, patch_bias)
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        # 解耦网络
        self.use_decoupled_module = use_decoupled_module
        if use_decoupled_module:
            self.decouple_module = Decoupled_Module(max_text_seq_length, text_embed_dim, activation=decoupled_activation)

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed_TI2V(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,  # latent w
            sample_height=sample_height,  # latent h
            sample_frames=sample_frames,  # pixel f
            temporal_compression_ratio=temporal_compression_ratio,  # 时间压缩
            max_text_seq_length=max_text_seq_length,  # max_squ_len
            spatial_interpolation_scale=spatial_interpolation_scale,  # 空间插值
            temporal_interpolation_scale=temporal_interpolation_scale,  # 时间插值
            use_positional_embeddings=not use_rotary_positional_embeddings,  # 使用位置编码还是RoPE编码
            use_learned_positional_embeddings=use_learned_positional_embeddings,  # 使用可学习的位置编码

            image_embedding_dim=image_embedding_dim,
            max_image_seq_length=max_image_seq_length
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)
        # [b, embed_dim]
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock_TI2V(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,

                    dropout=dropout,
                    activation_fn=activation_fn,  # 激活函数
                    attention_bias=attention_bias,  # attn b
                    norm_elementwise_affine=norm_elementwise_affine,  # 归一化
                    norm_eps=norm_eps,  # 归一化
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels
        # proj_out
        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,                # [b,f,c,h,w]
        encoder_hidden_states: torch.Tensor,        # [b,seq_len,text_embed_dim]
        timestep: Union[int, float, torch.LongTensor],  # [b,]
        encoder_hidden_states_image: Optional[torch.Tensor] = None,  # [b,seq_len,img_embed_dim]

        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,   # RoPE emb [torch.Tensor([num_patch, head_dim]), torch.Tensor([num_patch, head_dim])]
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True
    ):
        """
        Args:
            hidden_states:              [b,f,c,h,w]
            encoder_hidden_states:      [b,seq_len,text_embed_dim]
            timestep:                   [b,]
            timestep_cond:
            image_rotary_emb:           RoPE emb
            return_dict:

        Returns:
           Transformer2DModelOutput(sample=output)      output: [b,f,c,h,w]
        """
        # Lora
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding     emb: [b, time_embed_dim]
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding:   hidden_states: [b,text_seq_len + num_patches,embed_dim]
        # 解耦
        if self.use_decoupled_module:
            encoder_hidden_states_image = self.decouple_module(encoder_hidden_states, encoder_hidden_states_image, False)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states, encoder_hidden_states_image)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        img_seq_length = encoder_hidden_states_image.shape[1]
        # encoder_hidden_states: [b,text_seq_len,embed_dim]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        encoder_hidden_states_image = hidden_states[:, text_seq_length: text_seq_length + img_seq_length]
        # hidden_states: [b,num_patches,embed_dim]
        hidden_states = hidden_states[:, text_seq_length + img_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states, encoder_hidden_states_image = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_image,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states, encoder_hidden_states_image = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        # 归一化
        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_image, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length + img_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify          out: [b,f,c,h,w]
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            # output: [b, patches_t, patches_h, patches_w, c, p_t, p, p]
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            # [b, patches_t, patches_h, patches_w, c, p_t, p, p] -> [b,f,c,h,w]
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


if __name__ == '__main__':
    embed_dim = 1920
    text_embed_dim = 4096
    max_text_seq_length = 226
    f, c, w, h = 49, 16, 90, 60
    latent_f = (f - 1) // 4 + 1
    time_embed_dim = 512

    video = torch.randn(2, latent_f, c, h, w)
    text_embedding = torch.randn(2, max_text_seq_length, text_embed_dim)
    img_embedding = torch.randn(2, max_text_seq_length, text_embed_dim)
    timestep = torch.randn(2, )
    attention_head_dim = 64
    image_rotary_emb = [torch.randn(latent_f * w * h // 4, attention_head_dim), torch.randn(latent_f * w * h // 4, attention_head_dim)]
    # cog_patch_i2v = CogVideoXPatchEmbed_TI2V(2, in_channels= 16, embed_dim=embed_dim, text_embed_dim=text_embed_dim,
    #         sample_width=w, sample_height=h, sample_frames=f, temporal_compression_ratio=4, max_text_seq_length=max_text_seq_length, use_positional_embeddings=True,
    #         image_embedding_dim=text_embed_dim, max_image_seq_length=max_text_seq_length)
    # embeds = cog_patch_i2v(text_embedding, video, image_embeds=img_embedding)
    # print(embeds.shape)

    cogvideo_ti2v = CogVideoXTransformer3D_TI2V(
        num_attention_heads=30, attention_head_dim=64,
        in_channels=c, sample_width=w, sample_height=h, sample_frames=f, patch_size=2, temporal_compression_ratio=4,
        text_embed_dim=text_embed_dim, max_text_seq_length=max_text_seq_length,
        image_embedding_dim=text_embed_dim, max_image_seq_length=max_text_seq_length,
        time_embed_dim=time_embed_dim,
        num_layers=2, use_rotary_positional_embeddings=True, use_learned_positional_embeddings=True)

    pred = cogvideo_ti2v(hidden_states=video, encoder_hidden_states=text_embedding, timestep=timestep,
        encoder_hidden_states_image=img_embedding, image_rotary_emb=image_rotary_emb)
    print(pred.sample.shape)

