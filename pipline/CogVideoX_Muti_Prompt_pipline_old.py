import sys, os

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
import os

from omegaconf import OmegaConf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import CogVideoXPipeline
from transformers import T5EncoderModel, AutoTokenizer, CLIPVisionModel, CLIPProcessor
from transformers import CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers import CogVideoXPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import CogVideoXLoraLoaderMixin
from diffusers.models.embeddings import get_3d_rotary_pos_embed

from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

from Mutiple_prompt_mutiple_scene.Model.CogVideoX_ti2v import CogVideoXTransformer3D_TI2V
from Mutiple_prompt_mutiple_scene.Model.transformer_blocks import Feature_Extraction_Module
from Mutiple_prompt_mutiple_scene.Mydataset.data_utils import video2image, encode_video, encode_text
from Mutiple_prompt_mutiple_scene.utils import load_feature_extractor_from_ckpt, load_transformer_multi_ti2v

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from CogVideoX_Muti_Prompt_pipline import *

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipline = CogVideoX_MultiPrompt_Pipeline(tokenizer=tokenizer, text_encoder=text_encoder, image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, scheduler=scheduler, torch_dtype=torch.float16)
        >>> pipline = pipline.to('cpu')
        >>> prompt = 'The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene.'
        >>> past_prompts = ['The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 is holding the bag, and the bag is filled with objects. Actor1 opens the bag and says, "Hey everybody, Nick here, and today I got a review for you of this little guy. Um, this is the silent pocket uh 20-liter pack. Um first off though, I want to thank very much Silent...',
        ...                 'The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 says "of course, here it is against your...']
        >>> past_videos = [r'E:/PythonLearn/work/SSH_Connect/Autodl/under2postgraudate/Video-Generation-field/Ours/Multiple scene/excluded_dir/dataset/videos/0-ggn3z52oU_76/split/The Silent Pocket 20 Liter Faraday Pack A Quick Shabazz Review-Scene-002.mp4',
        ...                 r'E:\PythonLearn\work\SSH_Connect\Autodl/under2postgraudate/Video-Generation-field/Ours/Multiple scene/excluded_dir/dataset/videos/0-ggn3z52oU_76/split/The Silent Pocket 20 Liter Faraday Pack A Quick Shabazz Review-Scene-004.mp4']
        >>> output = pipline(prompt=prompt, past_prompts=past_prompts, past_images=past_videos,
        ...                     height=480, width=720, num_frames=49,
        ...                     num_inference_steps=50, guidance_scale=6.0, image_guidance_scale=6.0)
        >>> video = output.frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CogVideoX_MultiPrompt_Pipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    """
       CogVideoXPipline_Muti_Prompt的8个组件:
       1. text tokenizer
       2. text encoder
       3. image processor
       4. image encoder
       5. feature extractor
       6. vae
       7. transformer
       8. scheduler
       """
    _optional_components = []
    model_cpu_offload_seq = "text_encoder->image_encoder->feature_extractor->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "current_image_features",
        "current_image_features_without_past"
    ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 调用父类的 from_pretrained 方法加载预训练模型
        feature_extractor_path = kwargs.pop('feature_extractor_path')
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 加载 feature_extractor 的权重（如果提供）
        if feature_extractor_path:
            feature_extractor = load_feature_extractor_from_ckpt(os.path.join(feature_extractor_path, 'config.yaml'),
                                                                 os.path.join(feature_extractor_path, 'feature_extraction_model_last.pth'))
            pipeline.feature_extractor = feature_extractor

        return pipeline

    def __init__(
            self,
            tokenizer: AutoTokenizer = None,
            text_encoder: T5EncoderModel = None,
            image_encoder: CLIPVisionModelWithProjection = None,
            feature_extractor: Feature_Extraction_Module = None,
            vae: AutoencoderKLCogVideoX = None,
            transformer: CogVideoXTransformer3D_TI2V = None,
            scheduler: Optional[Union[CogVideoXDPMScheduler, CogVideoXDDIMScheduler]] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler,
            image_encoder=image_encoder, feature_extractor=feature_extractor
        )
        # vae空间压缩率
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # vae时间压缩率
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        # vae图像压缩率
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_videos_per_prompt: int = 1,
            max_sequence_length: int = 226,  # 最大序列长度
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        """
        对prompt编码为prompt embedding

        Args:
            prompt:
            num_videos_per_prompt:          num videos per prompt
            max_sequence_length:            max squ_len
        Returns:
            prompt embedding: [b,squ_len,dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            do_classifier_free_guidance: bool = True,
            num_videos_per_prompt: int = 1,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        r"""
        编码prompt和neg_prompt

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        Returns:
            prompt_embeds: [b,squ_len,dim], negative_prompt_embeds: [b,squ_len,dim]
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:  # encode prompt
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:  # encode neg_prompt
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def _get_image_embeds(self,
                          images: Union[List[str], str] = None,
                          device: Optional[torch.device] = None,
                          dtype: Optional[torch.dtype] = None, ):
        """
        :param images:                      [path/PIL.Image] or [tensor(c,h,w,uint8)]
        :param num_videos_per_prompt
        :param device:
        :param dtype:
        :return:
            [f,l,dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.image_encoder.dtype
        images = images if isinstance(images, list) else [images]

        image_features = encode_video(images, self.image_encoder, num_frames=49, device=device, dtype=dtype)
        return image_features

    def _get_image_feature(self,
                           current_prompt: str,
                           past_prompt: Optional[List[str]] = None,
                           past_image: Optional[List[str]] = None,
                           device: Optional[torch.device] = None,
                           dtype: Optional[torch.dtype] = None
                           ):
        """
        :param current_prompt:
        :param past_prompt:
        :param past_images:
        :param device:
        :param dtype:
        :return:
            [l,dim]
        """
        past_prompt_embeds, past_image_embeds = [], []
        if past_prompt is not None and len(past_prompt) != 0:
            past_prompt_embeds = self._get_t5_prompt_embeds(past_prompt, device=device, dtype=dtype)
        if past_image is not None and len(past_image) != 0:
            past_image_embeds = self._get_image_embeds(past_image, device=device, dtype=dtype)
        current_prompt_embeds = self._get_t5_prompt_embeds([current_prompt], device=device, dtype=dtype)
        current_image_feature = self.feature_extractor([past_image_embeds], [past_prompt_embeds], current_prompt_embeds)
        current_image_feature = current_image_feature[0].to(dtype=dtype, device=device)
        return current_image_feature

    def get_image_features(self,
                           current_prompts: Union[List[str]] = None,
                           past_prompts: Union[List[List[str]]] = None,
                           past_images: Union[List[List[str]]] = None,
                           current_image_features: Optional[torch.Tensor] = None,
                           do_image_feature_classifier_free_guidance=True,
                           current_image_features_without_past=None,
                           device: Optional[torch.device] = None,
                           dtype: Optional[torch.dtype] = None,
                           ):
        """
        :param current_prompt:          [b,]
        :param past_prompt:             [[],[1,],[2,],[3]]
        :param past_images:             [[],[1,],[2,],[3]]

        :param current_image_features:    [b,l,dim]
        :param device:
        :param dtype:
        :return:
            current_image_features:[b,l,dim]
            current_image_features_without_past: [b,l,dim] or []

        批量 处理特征抽取
        classfier-free-guidance:
            epsilon(z_t,t,P,{P_past,I_past})=epsilon(z_t,t,P,None,None)+w*(epsilon(z_t,t,P,{P_past,I_past})-epsilon(z_t,t,P,None,None))
        """
        device = device or self._execution_device
        dtype = dtype or self.image_encoder.dtype

        # 收取视频特征
        if current_image_features is not None:
            current_image_features = current_image_features.to(dtype=dtype, device=device)
        else:
            assert current_prompts is not None, f'current_prompts不能为None'
            current_image_features = []
            for i in range(len(current_prompts)):
                past_prompt = past_prompts[i] if past_prompts is not None else []
                past_image = past_images[i] if past_images is not None else []
                current_image_features.append(
                    self._get_image_feature(current_prompts[i], past_prompt, past_image, device, dtype))

            current_image_features = torch.stack(current_image_features, dim=0).to(dtype=dtype, device=device)

        # Classifier-free guidance
        if not do_image_feature_classifier_free_guidance:
            return current_image_features, None

        current_image_features_without_past = current_image_features_without_past or []
        if do_image_feature_classifier_free_guidance and current_image_features_without_past == []:
            for i in range(len(current_prompts)):
                current_image_features_without_past.append(
                    self._get_image_feature(current_prompts[i], [], [], device, dtype))

        if isinstance(current_image_features_without_past, list) and len(current_image_features_without_past) != 0:
            current_image_features_without_past = torch.stack(current_image_features_without_past, dim=0).to(
                dtype=dtype, device=device)
        elif isinstance(current_image_features_without_past, torch.Tensor):
            current_image_features_without_past = current_image_features_without_past.to(device=device, dtype=dtype)

        return current_image_features, current_image_features_without_past

    def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        """
        Args:
            [b,c,f,h,w]: video像素空间的shape
        Returns:
            latents: [b, (f-1)/4, c, h/8, w/8]
        """
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents:    [b, f, c, h, w]
        Returns:
            frames: [b, c, f, h, w]
        """
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
            self,
            prompt,
            height,
            width,

            past_prompts,
            past_images,

            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            current_image_features=None,
            current_image_features_without_past=None
    ):
        """检查输入"""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        # image features check
        should_pred_image_features = prompt is not None
        if should_pred_image_features and ((past_images is None) != (past_prompts is None)):
            raise ValueError('past_images和past_prompts应该同时存在或同时为None')
        if should_pred_image_features and past_images is not None and past_prompts is not None:
            if len(past_images) != len(past_prompts):
                raise ValueError('past_images和past_prompts的batch_size应该相同')
        if not should_pred_image_features and current_image_features is None:
            raise ValueError('current_prompts,past_prompts,past_images不能为空或者current_image_features不能为空')
        if should_pred_image_features and current_image_features is not None:
            raise ValueError(
                'current_prompt,past_images,past_prompts和current_image_features都有值,只能前3个有值或current_image_features有值')
        if current_image_features is not None and current_image_features_without_past is not None:
            if current_image_features.shape != current_image_features_without_past.shape:
                raise ValueError(
                    f'current_image_features.shape:{current_image_features.shape} != current_image_features_without_past.shape: {current_image_features_without_past.shape}')

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
            self,
            height: int,
            width: int,
            num_frames: int,
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D RoPE

        Args:
            height:             video的h
            width:              video的w
            num_frames:         video的f
        Returns:
            cos,sin: [base_num_frames * grid_height * grid_width, attention_head_dim / 2]
        """
        # 空间分块
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        # transformer默认的分块数
        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            # 时间分块
            base_num_frames = (num_frames + p_t - 1) // p_t
            # cos,sin: [base_num_frames * grid_height * grid_width, attention_head_dim / 2]
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,

            past_prompts: Optional[Union[List[str], List[List[str]]]] = None,
            past_images: Optional[Union[List[str], List[List[str]]]] = None,

            height: Optional[int] = None,
            width: Optional[int] = None,
            num_frames: Optional[int] = None,

            num_inference_steps: int = 50,  # DDIM inference steps
            timesteps: Optional[List[int]] = None,  # DDIM timesteps set
            guidance_scale: float = 6,  # Classifier-free guidance
            image_guidance_scale: float = 6,  # image Classifier-free guidance

            use_dynamic_cfg: bool = False,
            num_videos_per_prompt: int = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            current_image_features: Optional[torch.FloatTensor] = None,
            current_image_features_without_past: Optional[torch.FloatTensor] = None,

            output_type: str = "pil",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 在每个时间步去噪最后执行: callback_on_step_end(self, i, t, callback_kwargs), 并返回latents,prompt_embeds,negative_prompt_embeds
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 226,  # text embedding squ_len
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Classifier-free guidance:
            \epsilon(z_t,t,P+,{P+,I})=\epsilon(z_t,t,P-,{P+,I})+w1*(\epsilon(z_t,t,P+,{P+,I})-\epsilon(z_t,t,P-,{P+,I}))+
            w2*(\epsilon(z_t,t,P+,{P+,I})-\epsilon(z_t,t,P+,{None,None}))
            其中k随机去除一些{P,I}_k集合，让模型学习到即使不需要过去信息也能生成对应的图像
            TODO: k随机取

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.

            CogVideoXPipelineOutput.frames
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        if prompt is not None and isinstance(prompt, str):
            prompt = [prompt]
        if past_prompts is not None and isinstance(past_prompts, list) and len(past_prompts) != 0 and not isinstance(
                past_prompts[0], list):
            past_prompts = [past_prompts]
        if past_images is not None and isinstance(past_images, list) and len(past_images) != 0 and not isinstance(
                past_images[0], list):
            past_images = [past_images]
        if current_image_features is not None and current_image_features.ndim == 2:
            current_image_features = current_image_features.unsqueeze(0)
        if current_image_features_without_past is not None and current_image_features_without_past.ndim == 2:
            current_image_features_without_past = current_image_features_without_past.unsqueeze(0)
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,

            past_prompts,
            past_images,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
            current_image_features,
            current_image_features_without_past
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # image classifier-free guidance
        do_image_classifier_free_guidance = image_guidance_scale > 1.0

        # 3. Encode input prompt
        # prompt_embeds, negative_prompt_embeds: [b,squ_len,dim]
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        current_image_features, current_image_features_without_past = self.get_image_features(
            prompt, past_prompts, past_images, current_image_features,
            do_image_classifier_free_guidance, current_image_features_without_past, device)

        # Classifier-free guidance concat: 在batch_size维度拼接
        prompt_embeds_tmp = prompt_embeds.clone()
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            current_image_features = torch.cat([current_image_features, current_image_features], dim=0)
        if do_image_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_tmp], dim=0)
            current_image_features = torch.cat([current_image_features, current_image_features_without_past], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents: [b,f,c,h,w]
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        # 时间分块需要填充
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        # image_rotary_emb: (cos,sin): 2 * [base_num_frames * grid_height * grid_width, attention_head_dim / 2]
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if do_classifier_free_guidance and do_image_classifier_free_guidance:
                    mult = 3
                elif do_classifier_free_guidance or do_image_classifier_free_guidance:
                    mult = 2
                else:
                    mult = 1

                latent_model_input = torch.cat([latents] * mult, dim=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,  # [latent,latent] latent: [b,f,c,h,w]
                    encoder_hidden_states=prompt_embeds,  # [prompt_embedding, neg_prompt_embedding]
                    encoder_hidden_states_image=current_image_features,  # [2*b,l,dim]
                    timestep=timestep,  # t
                    image_rotary_emb=image_rotary_emb,  # 3D RoPE
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(
                                math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                    self._image_guidance_scale = 1 + image_guidance_scale * (
                            (1 - math.cos(
                                math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance and do_image_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text, noise_pred_text_without_past = noise_pred.chunk(3)
                    noise_pred = (noise_pred_uncond +
                                  self.guidance_scale * (noise_pred_text - noise_pred_uncond) +
                                  self._image_guidance_scale * (noise_pred_text - noise_pred_text_without_past))
                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif do_image_classifier_free_guidance:
                    noise_pred_text, noise_pred_text_without_past = noise_pred.chunk(2)
                    noise_pred = noise_pred_text_without_past + self._image_guidance_scale * (
                                noise_pred_text - noise_pred_text_without_past)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Return
        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            # 填充时会填充第一帧，丢弃时丢弃前additional_frames帧
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


if __name__ == '__main__':
    model_dir = '../../excluded_dir/local_model/model_dir'
    # 加载所有组件
    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(
        model_dir, subfolder="text_encoder", local_files_only=True
    )
    # image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_dir, subfolder='Clip4Clip',
                                                                  local_files_only=True)

    configs = OmegaConf.load(os.path.join(model_dir, 'feature_extractor', 'config.yaml'))
    feature_extractor = load_feature_extractor_from_ckpt(configs, os.path.join(model_dir, 'feature_extractor',
                                                                               'feature_extraction_model_last.pth'))
    transformer = load_transformer_multi_ti2v(model_dir, subfolder='transformer', local_files_only=True)
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_dir, subfolder="vae", local_files_only=True)

    print('加载所有组件完成')
    pipline = CogVideoX_MultiPrompt_Pipeline(tokenizer=tokenizer, text_encoder=text_encoder,
                                             image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae,
                                             scheduler=scheduler, torch_dtype=torch.float16)
    pipline = pipline.to('cpu')

    prompt = 'The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene.'
    past_prompts = [
        'The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 is holding the bag, and the bag is filled with objects. Actor1 opens the bag and says, "Hey everybody, Nick here, and today I got a review for you of this little guy. Um, this is the silent pocket uh 20-liter pack. Um first off though, I want to thank very much Silent...',
        'The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 says "of course, here it is against your...']
    past_videos = [
        r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\excluded_dir\dataset\videos\0-ggn3z52oU_76\split\The Silent Pocket 20 Liter Faraday Pack A Quick Shabazz Review-Scene-002.mp4',
        r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\excluded_dir\dataset\videos\0-ggn3z52oU_76\split\The Silent Pocket 20 Liter Faraday Pack A Quick Shabazz Review-Scene-004.mp4']
    output = pipline(prompt=prompt, past_prompts=past_prompts, past_images=past_videos,
                     height=480, width=720, num_frames=49,
                     num_inference_steps=50, guidance_scale=6.0, image_guidance_scale=6.0)


