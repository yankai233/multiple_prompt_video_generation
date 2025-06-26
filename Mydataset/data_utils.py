from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import cv2
import numpy as np
import torch
from typing import List, Union, Optional, Dict
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor, T5Tokenizer, CLIPTextModelWithProjection
from transformers import T5EncoderModel
from transformers import CLIPVisionModelWithProjection
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


# Code to convert one video to few images.
def video2image(video_path, target_frames=49, size=224):
    def preprocess(size, image):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(image)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"ERROR: Unable to open video file: {video_path}")
        return torch.zeros([target_frames, 3, size, size], dtype=torch.float32)

    # 获取视频的总帧数和帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        print(f"ERROR: Invalid frame rate ({fps}) in video file: {video_path}")
        cap.release()
        return torch.zeros([target_frames, 3, size, size], dtype=torch.float32)

    # 计算需要提取的帧索引
    if frame_count < target_frames:
        # 如果视频帧数不足，则填充最后一帧
        frame_indices = np.arange(0, frame_count)
        padding_frames = target_frames - frame_count
    else:
        # 如果视频帧数足够，则均匀采样
        frame_indices = np.linspace(0, frame_count - 1, target_frames, dtype=int)

    # 初始化输出张量
    images = np.zeros([target_frames, 3, size, size], dtype=np.float32)
    last_frame = None

    # 逐帧读取视频
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        images[i] = preprocess(size, Image.fromarray(frame))

    # 如果帧数不足，用最后一帧填充
    if frame_count < target_frames:
        for i in range(frame_count, target_frames):
            images[i] = preprocess(size, Image.fromarray(last_frame))

    # 释放视频文件
    cap.release()

    # 返回帧张量
    return torch.tensor(images)


def encode_text(prompt, tokenizer: Union[CLIPTokenizer, T5Tokenizer], text_encoder: Union[CLIPTextModelWithProjection, T5EncoderModel]):
    """
    将prompt编码为embedding
    :param prompt:   str, [str]
    :return:
        [b,77,dim]
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    if isinstance(tokenizer, CLIPTokenizer):
        inputs = tokenizer(text=prompt, return_tensors="pt", padding='max_length', truncation=True,
                           max_length=77)
        outputs = text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state

        # Normalize embeddings for retrieval:
        text_embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
    else:
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(
            text_input.input_ids.to(text_encoder.device)
        )[0]
    return text_embeddings


def encode_video(video_batch: List[str], video_encoder, video_size=224, num_frames=49, device=None, dtype=None) -> torch.Tensor:
    """批量编码video -> video embedding"""
    # TODO: 读取视频
    video_batch = [video_batch] if isinstance(video_batch, str) else video_batch
    device = device or video_encoder.device
    dtype = dtype or video_encoder.dtype
    video_encoder = video_encoder.eval()
    video_embeddings = []
    for video_path in video_batch:
        video = video2image(video_path, target_frames=num_frames, size=video_size)
        vision_output = video_encoder(video.to(device=device, dtype=dtype))
        visual_output = vision_output["image_embeds"]
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_embeddings.append(visual_output)
    video_embeddings = torch.stack(video_embeddings, dim=0)
    return video_embeddings

