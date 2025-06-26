"""
提取语义模组: 根据<V_i,P_i>提取语义
    1. 将<V_i,P_i> -> <I_i,P_i> -> <image_embedding_i, text_embedding_i>
    2. 线性映射+交叉注意力: Q = W_Q(linear(text_embedding_i)), K,V=W_QK(linear(image_embedding_i))
        计算得到注意力: Attn_i    [1,77,512]   与text_embedding_i拼接为[1,77,512]
    3. 将[Attn_i,text_embedding_i]注入解耦网络解耦(线性层), 拆分为image_feature_i,text_feature_i
    4. 将image_feature_i,text_feature_i注入CogVideoX
"""

import torch
import torch.nn as nn  # 网络
import torch.nn.functional as F  # 激活函数
import torchvision.datasets  # 样本
from pandas.io.formats.format import return_docstring
from torch.utils import *  # 图案管理
from torch.utils import data  # 预处理数据集
from torchvision import transforms  # 工具
import torch.optim as optim
from tqdm import tqdm  # 进度条
import numpy as np
import matplotlib.pyplot as plt
import os
from diffusers.models.attention import Attention
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from transformers import CLIPVisionModelWithProjection


activation_fn = {
    'gelu': nn.GELU(),
    'relu': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softmax': nn.Softmax(),
    }


class Feature_Extraction_Module(nn.Module):
    """特征抽取模组"""
    def __init__(self,
                 text_squ_len=226,               # 输出的token数量(tokenizer的输出token数量)
                 text_embed_dim=4096,
                 image_embed_dim=768,

                 proj_dim=4096,                  # hidden_dim
                 cross_attn_heads=32,            # cross attn heads
                 cross_attn_head_dim=128,        # cross attn dim
                 cross_attn_bias=True,          # cross attn b
                 qk_norm='layer_norm',          # cross attn norm
                 cross_attn_eps=1e-6,                      # cross attn eps
                 cross_attn_out_bias=True,      # cross attn out b
                 activation='gelu',             # 激活函数

                 # transformer_dim=1024,         # transformer dim
                 transformer_model='CLIP-ViT',   # ['CLIP-ViT', 'TransformerEncoderLayer']

                 clip_model_id = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',   # CLIP-ViT的路径

                 transformer_head=32,            # transformer head
                 transformer_ff_dim=16384,       # transformer FeedForward
                 transformer_activation='gelu', # 激活函数
                 transformer_eps=1e-6,          # eps

                 return_text_feature=False,      # 是否返回提取的文本信息
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.proj_dim = proj_dim
        self.activation = activation_fn[activation]
        self.image_proj = nn.Linear(image_embed_dim, proj_dim)
        self.text_proj = nn.Linear(text_embed_dim, proj_dim)
        self.image_norm1 = nn.LayerNorm(self.proj_dim)
        self.text_norm1 = nn.LayerNorm(self.proj_dim)
        self.cross_attn = Attention(query_dim=proj_dim, heads=cross_attn_heads, dim_head=cross_attn_head_dim,
                                    bias=cross_attn_bias,
                                    qk_norm=qk_norm if qk_norm else None, eps=cross_attn_eps,
                                    out_bias=cross_attn_out_bias)
        self.output_norm = nn.LayerNorm(self.proj_dim)

        self.return_text_feature = return_text_feature
        # 从开始的可学习的向量: [77, proj_dim]
        self.text_squ_len = text_squ_len
        self.learnable_feature = nn.Parameter(torch.zeros(text_squ_len, proj_dim))
        # transformer block
        self.transformer_model = transformer_model
        if transformer_model == 'TransformerEncoderLayer':
            self.transformer_block = nn.TransformerEncoderLayer(
                d_model=2 * self.proj_dim, nhead=transformer_head, dim_feedforward=transformer_ff_dim,
                activation=transformer_activation, layer_norm_eps=transformer_eps
            )
        elif transformer_model == 'CLIP-ViT':
            self.transformer_input_linear = nn.Linear(2 * proj_dim, 1280)
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_id, local_files_only=True, use_safetensors=True)
            self.transformer_output_linear = nn.Linear(1024, 2 * proj_dim)

    def forward(self, image_embeddings, text_embeddings, current_text_embeddings):
        """
        :param text_embeddings: list(tensor)   [[1,L1,512], [2,L1,512], [3,L1,512], [4,L1,512]]
        :param image_embeddings:  list(tensor)   [[1,L2,768], [2,L2,768], [3,L2,768], [4,L2,768]]
        :param current_text_embeddings: list(tensor) [[L1,512], [L1,512], [L1,512], [L1,512]]
        :return:
            [b,77,512]
        for batch in batchs:
            1. 将image_embeddings和text_embedding输入线性层
            2. 同b和f的text和image做交叉注意力, attn: [b,f,77,512]
            3. attn和text_emebdding拼接
            4. 输入transformer block,取前77个token得到attn_pred

        训练时：
        1. 计算当前帧图像和文本(经过线性层)的交叉注意力 attn_real
        2. 计算L=||attn_real, attn_pred||^2

        之后输入给解耦网络
        """
        batch_map_output = []
        text_output = []
        text_squ_len = current_text_embeddings[0].shape[-2]
        assert text_squ_len == self.text_squ_len, f'输入的text_squ_len与实际的text_squ_len不同，请确保使用相同的tokenizer和text_encoder模型'
        assert len(image_embeddings) == len(text_embeddings), (f'batch必须相同, 但你的image_embeddings.batch={len(image_embeddings)}, '
                                                        f'text_embeddings.batch={len(text_embeddings)}')

        # 遍历每个batch
        for i in range(len(current_text_embeddings)):
            # [f,l2,768], [f,l1,512]
            current_text_embedding = current_text_embeddings[i]
            current_text_embedding = self.text_norm1(self.activation(self.text_proj(current_text_embedding)))
            if len(image_embeddings) != 0 and len(text_embeddings) != 0 and image_embeddings[i] != [] and text_embeddings[i] != []:
                image_embedding = image_embeddings[i]
                text_embedding = text_embeddings[i]
                # 1. 线性映射
                image_feature = self.image_norm1(self.activation(self.image_proj(image_embedding)))
                text_feature = self.text_norm1(self.activation(self.text_proj(text_embedding)))

                # 2. 交叉注意力: [f,l1,4096]
                cross_attn = self.cross_attn(hidden_states=text_feature, encoder_hidden_states=image_feature)


                # 3. 拼接: [attn,text]
                concat_attn_text = torch.cat([cross_attn, text_feature], dim=-1)
                concat_attn_text = concat_attn_text.reshape(-1, 2 * self.proj_dim)
                #    拼接: [[learnable_feature, current_text], cross_attn]
                concat_current_attn_text = torch.cat([self.learnable_feature, current_text_embedding], dim=-1)
                concat_attn_text = torch.cat([concat_current_attn_text, concat_attn_text], dim=-2)
            else:
                concat_attn_text = torch.cat([self.learnable_feature, current_text_embedding], dim=-1)

            # 4. 注入transformer block
            # [1, f * l1, 4096 * 2]
            concat_attn_text = concat_attn_text[None, :, :]
            if self.transformer_model == 'TransformerEncoderLayer':
                extract_feature = self.transformer_block(concat_attn_text)[0]
                extract_feature = extract_feature[:text_squ_len]
            elif self.transformer_model == 'CLIP-ViT':
                embeddings = self.transformer_input_linear(concat_attn_text)
                embeddings = self.image_encoder.vision_model.pre_layrnorm(embeddings)
                encoder_outputs = self.image_encoder.vision_model.encoder(
                    inputs_embeds=embeddings,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None)
                last_hidden_state = encoder_outputs[0]
                # 取出style embedding f_s [B,text_squ_len,1280]
                pooled_output = last_hidden_state[:, :text_squ_len, :]
                # 层归一化  pooled_output:[3,1280]
                pooled_output = self.image_encoder.vision_model.post_layernorm(pooled_output)
                extract_feature = self.image_encoder.visual_projection(pooled_output)[0]

                extract_feature = self.transformer_output_linear(extract_feature)
            else:
                raise ValueError(f'transformer_model只能为: TransformerEncoderLayer和CLIP-ViT, 你的transformer_model: {self.transformer_model}')
            # 分割 + 归一化
            extract_cross_map_feature, extract_text_feature = extract_feature.chunk(2, dim=-1)
            extract_cross_map_feature = self.output_norm(extract_cross_map_feature)
            extract_text_feature = self.output_norm(extract_text_feature)

            batch_map_output.append(extract_cross_map_feature)
            if self.return_text_feature:
                text_output.append(extract_text_feature)

        batch_map_output = torch.stack(batch_map_output, dim=0)
        if self.return_text_feature:
            text_output = torch.stack(text_output, dim=0)
            return batch_map_output, text_output
        return batch_map_output

    @torch.no_grad()
    def no_transformer(self, current_image_embeddings, current_text_embeddings):
        """
        只计算交叉注意力: attn_real, 用于计算损失函数
        :param image_embeddings:   [b,L2,768]
        :param text_embeddings:    [b,L1,512]
        :return:
            [b,77,512]
        """
        if isinstance(current_text_embeddings, list):
            current_text_embeddings = torch.stack(current_text_embeddings, dim=0)
        if isinstance(current_image_embeddings, list):
            current_image_embeddings = torch.stack(current_image_embeddings, dim=0)

        text_squ_len = current_text_embeddings[0].shape[-2]
        assert text_squ_len == self.text_squ_len, f'输入的text_squ_len与实际的text_squ_len不同，请确保使用相同的tokenizer和text_encoder模型'
        assert len(current_image_embeddings) == len(current_text_embeddings), (
            f'batch必须相同, current_image_embeddings.batch={len(current_image_embeddings)}, '
            f'current_text_embeddings.batch={len(current_text_embeddings)}')

        # 1. 线性映射
        image_feature = self.image_norm1(self.activation(self.image_proj(current_image_embeddings)))
        text_feature = self.text_norm1(self.activation(self.text_proj(current_text_embeddings)))

        # 2. 交叉注意力: [f,l1,512]
        cross_attn = self.cross_attn(hidden_states=text_feature, encoder_hidden_states=image_feature)
        output = self.output_norm(cross_attn)
        return output


if __name__ == '__main__':
    image_embeddings = [torch.randn(1, 88, 768), torch.randn(2, 88, 768)]
    text_embeddings = [torch.randn(1, 77, 4096), torch.randn(2, 77, 4096)]
    # image_embeddings = []
    # text_embeddings = []
    current_text_embeddings = [torch.randn(226, 4096), torch.randn(226, 4096)]
    model = Feature_Extraction_Module(text_squ_len=226, text_embed_dim=4096, proj_dim=512, cross_attn_heads=8, cross_attn_head_dim=64, transformer_head=8, transformer_ff_dim=512 * 4)
    pred = model(image_embeddings, text_embeddings, current_text_embeddings)
    print(pred.shape)

