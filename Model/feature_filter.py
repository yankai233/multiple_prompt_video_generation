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
from copy import deepcopy


def feature_filter(prompt_embeddings, current_prompt_embeddings,
                   filter_past_token_indexs, filter_current_token_indexs,
                   padding_embedding:torch.Tensor, max_suq_len=226
    ):
    """
    :param prompt_embeddings:               [[f-1,l,dim]]
    :param current_prompt_embeddings:       [b,l,dim]
    :param past_prompts:                    [['1', '2']]
    :param current_prompts:                 ['1','2']
    :param filter_past_prompts:             [['f1','f2']]
    :param filter_current_prompts:          ['f1','f2']
    :param filter_past_token_indexs:        [[[1,2],[1,2]]]
    :param filter_current_token_indexs:     [[1,2]]
    :param padding_embedding:               [dim,]
    :param max_suq_len:                      226
    :return:
        filter_prompt_embeddings:                       [[f-1,l,dim]]
        filter_current_prompt_embeddings:               [b,l,dim]

    past_prompts  ->  prompt_embeddings
    current_prompts  ->  current_prompt_embeddings
    filter_past_prompts -> filter_past_prompt_embeddings
    filter_current_prompts -> filter_current_prompt_embeddings
    """

    # 处理筛选索引，移除 -1 的元素
    tmp_filter_past_token_indexs = [
        [[i for i in scene if i != -1] for scene in batch] if batch else []
        for batch in filter_past_token_indexs
    ]
    tmp_filter_current_token_indexs = [
        [i for i in batch if i != -1] if batch else []
        for batch in filter_current_token_indexs
    ]

    # 特征筛选
    filter_past_prompt_embeddings, filter_current_prompt_embeddings = [], []
    for b in range(len(prompt_embeddings)):
        prompt_embedding = prompt_embeddings[b]
        current_prompt_embedding = current_prompt_embeddings[b]

        filter_past_token_index = tmp_filter_past_token_indexs[b]
        filter_current_token_index = tmp_filter_current_token_indexs[b]

        filter_past_prompt_embedding, filter_current_prompt_embedding = [], []
        if prompt_embedding is None or len(prompt_embedding) == 0:
            filter_past_prompt_embeddings.append([])
        else:
            for scene in range(len(prompt_embedding)):
                # [l,dim]
                scene_prompt_embedding = prompt_embedding[scene]
                scene_past_token_index = filter_past_token_index[scene]
                if len(scene_past_token_index) == 0:
                    filter_scene_prompt_embedding = padding_embedding.repeat(max_suq_len, 1)
                else:
                    filter_scene_prompt_embedding = torch.cat([scene_prompt_embedding[scene_past_token_index], padding_embedding.repeat(max_suq_len - len(scene_past_token_index), 1)], dim=0)

                filter_past_prompt_embedding.append(filter_scene_prompt_embedding)
            filter_past_prompt_embeddings.append(torch.stack(filter_past_prompt_embedding, dim=0))

        # current_prompt
        assert len(filter_current_token_index) > 0, f'筛选后的current prompt的token数量要大于0'

        filter_scene_current_prompt_embedding = torch.cat(
            [current_prompt_embedding[filter_current_token_index],
             padding_embedding.repeat(max_suq_len - len(filter_current_token_index), 1)], dim=0)

        filter_current_prompt_embeddings.append(filter_scene_current_prompt_embedding)
    filter_current_prompt_embeddings = torch.stack(filter_current_prompt_embeddings, dim=0)
    return filter_current_prompt_embeddings, filter_past_prompt_embeddings


if __name__ == '__main__':
    prompt_embeddings = [[], torch.randn((1, 226, 4096)), torch.randn((2, 226, 4096))]
    current_prompt_embeddings = torch.randn(3, 226, 4096)
    filter_past_token_indexs = [[], [[1, 2]], [[2, 3], [1, 2]]]
    filter_current_token_indexs = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
    padding_embedding = torch.randn((4096,))
    filter_current_prompt_embeddings, filter_past_prompt_embeddings = feature_filter(prompt_embeddings, current_prompt_embeddings, filter_past_token_indexs,
                   filter_current_token_indexs, padding_embedding, max_suq_len=226)
    print(filter_past_prompt_embeddings)
    print(len(filter_past_prompt_embeddings))
    print(filter_past_prompt_embeddings[1].shape)
    print(filter_current_prompt_embeddings.shape)

