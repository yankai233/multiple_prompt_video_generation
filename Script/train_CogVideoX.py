"""
训练CogVideoX_TI2V:
    1. CogVideoXTransformer3D_TI2V.patch_embed.image_proj

    2. CogVideoXTransformer3D_TI2V.transformer_blocks[:].norm1.linear2
       CogVideoXTransformer3D_TI2V.transformer_blocks[:].norm2.linear2
       # CogVideoXTransformer3D_TI2V.transformer_blocks[:].attn1.W_Q
       CogVideoXTransformer3D_TI2V.transformer_blocks[:].attn1.W_k
       CogVideoXTransformer3D_TI2V.transformer_blocks[:].attn1.W_v

修改至: https://github.com/THUDM/CogVideo
"""
import glob
import sys
import os

import tqdm

sys.path.append(os.path.join(os.getcwd(), '../../'))


import re
from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from omegaconf import OmegaConf

from transformers import AutoTokenizer, T5EncoderModel, CLIPVisionModel, CLIPProcessor
from transformers import CLIPVisionModelWithProjection
from typing_extensions import override

from Mutiple_prompt_mutiple_scene.Model.CogVideoX_ti2v import CogVideoXTransformer3D_TI2V
from Mutiple_prompt_mutiple_scene.utils import *
from Mutiple_prompt_mutiple_scene.Script.finetune.trainer import *
from Mutiple_prompt_mutiple_scene.Script.finetune.schemas import Components
from Mutiple_prompt_mutiple_scene.Script.finetune.utils import unwrap_model
from Mutiple_prompt_mutiple_scene.Script.finetune.models.utils import register
from Mutiple_prompt_mutiple_scene.Model.CogVideoX_ti2v import CogVideoXTransformer3D_TI2V
from Mutiple_prompt_mutiple_scene.Model.transformer_blocks import Feature_Extraction_Module
# from Mutiple_prompt_mutiple_scene.pipline.CogVideoX_Muti_Prompt_pipline import CogVideoX_MultiPrompt_Pipeline
from Mutiple_prompt_mutiple_scene.Mydataset.data_utils import encode_text, encode_video

from pydantic import BaseModel
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
logger = get_logger(LOG_NAME, LOG_LEVEL)
train_layer_name_lst = []



def get_trainable_module_name(components: Components):
    """获取需要训练的层名"""
    transformer = components.transformer
    train_layer_name_lst = []

    attn_train_lst = ['to_k', 'to_v']  # 注意力层可为调: ['to_q', 'to_k', 'to_v', 'norm_q','norm_k','norm_v','to_out']
    for layer_name, states_params in transformer.state_dict().items():
        if layer_name.startswith('patch_embed.image_proj'):
            train_layer_name_lst.append(layer_name)
        elif layer_name.startswith('decouple_module'):
            train_layer_name_lst.append(layer_name)
        elif layer_name.startswith('transformer_blocks'):
            if re.search(r'transformer_blocks\.\d+\.norm1\.linear2', layer_name):
                train_layer_name_lst.append(layer_name)
            elif re.search(r'transformer_blocks\.\d+\.norm2\.linear2', layer_name):
                train_layer_name_lst.append(layer_name)
            elif re.search(r'transformer_blocks\.\d+\.attn1\.', layer_name):  # 注意力层
                for attn in attn_train_lst:
                    if attn in layer_name:
                        train_layer_name_lst.append(layer_name)
                        break
    logger.info(f'一下参数将被训练:{train_layer_name_lst}')
    return train_layer_name_lst


class CogVideoX_MultiTI2V_Trainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae", "image_encoder", "feature_extractor"]

    @override
    def load_components(self) -> Components:
        """加载模型组件:vae,tokenizer,text_encoder,transformer,scheduler"""
        print('加载组件中...')
        components = Components()
        model_path = get_path_str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        print('加载tokenizer完成')

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", local_files_only=True
        )
        print('加载text encoder完成')

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", local_files_only=True)
        print('加载vae完成')

        components.transformer = load_transformer_multi_ti2v(model_path, subfolder='transformer', local_files_only=True)
        print('加载transformer完成')

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler", local_files_only=True
        )
        print('加载scheduler完成')

        components.feature_extractor = load_feature_extractor_from_ckpt(self.Feature_Extraction_Module_config, self.args.feature_extractor_ckpt)
        print('加载feature_extractor完成')

        # image encoder
        components.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder='Clip4Clip', local_files_only=True)
        print('加载视频编码器完成')

        print('加载所有组件完成')

        return components

    def __init__(self, args: Args):
        self.configs = OmegaConf.load(args.feature_extractor_config)
        self.Feature_Extraction_Module_config = self.configs.model.Feature_Extraction_Module
        super().__init__(args)

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        samples:
        [{
            "prompt_embeddings": prompt_embedding_lst,              # [f-1,l,dim]
            "current_prompt_embedding": current_prompt_embedding,   # [l,dim]
            "image_embeddings": image_embedding_lst,                # [f-1,l,dim]
            "current_image_embedding": current_image_embedding,     # [l,dim]
            "current_video_latent": encoded_video,                  # [C,F,H,W]
            "current_image_feature": current_image_feature,         # [l,dim]

            "video_metadata": {
                "prompt_regress": prompt_regress,                   # [f,]
                "video_path_regress": video_regress,                # [f,]
                "num_frames": encoded_video.shape[1],               # F
                "height": encoded_video.shape[2],                   # H
                "width": encoded_video.shape[3],                    # W
            },
        },...]
        """
        # TODO: 集成特征筛选
        res = {'prompt_embeddings': [], 'current_prompt_embedding': [], 'image_embeddings': [], 'current_image_embedding': [],
               'current_video_latent': [], 'current_image_feature': [],
               'video_metadata': {'prompt_regress': [], 'video_path_regress': [], 'num_frames': [], 'height': [], 'width': []}}
        for sample in samples:
            res['prompt_embeddings'].append(sample['prompt_embeddings'])
            res['image_embeddings'].append(sample['image_embeddings'])

            res['current_prompt_embedding'].append(sample['current_prompt_embedding'])
            res['current_image_embedding'].append(sample['current_image_embedding'])

            res['current_video_latent'].append(sample['current_video_latent'])
            res['current_image_feature'].append(sample['current_image_feature'])

            res['video_metadata']['prompt_regress'].append(sample['video_metadata']['prompt_regress'])
            res['video_metadata']['video_path_regress'].append(sample['video_metadata']['video_path_regress'])
            res['video_metadata']['num_frames'].append(sample['video_metadata']['num_frames'])
            res['video_metadata']['height'].append(sample['video_metadata']['height'])
            res['video_metadata']['width'].append(sample['video_metadata']['width'])
        # stack
        res['current_prompt_embedding'] = torch.stack(res['current_prompt_embedding'], dim=0)
        res['current_image_embedding'] = torch.stack(res['current_image_embedding'], dim=0)
        res['current_video_latent'] = torch.stack(res['current_video_latent'], dim=0)
        res['current_image_feature'] = torch.stack(res['current_image_feature'], dim=0)
        return res

    @override
    def prepare_dataset(self) -> None:
        """准备数据集"""
        logger.info("Initializing dataset and dataloader")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.image_encoder.requires_grad_(False)
        self.components.feature_extractor.requires_grad_(False)

        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.image_encoder = self.components.image_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.feature_extractor = self.components.feature_extractor.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        self.dataset = MultiTI2VDataset(
           **(self.args.model_dump()),
            root_path=self.args.data_root,
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        # 第一次执行prompt编码和video编码操作，将结果存储在cache中
        for batch in tqdm(tmp_data_loader, desc='prepare dataset', disable=not self.accelerator.is_local_main_process):
            try:
                pass
            except Exception as e:
                logger.error(e)
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")
        # 释放vae和text_encoder空间
        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        unload_model(self.components.image_encoder)
        unload_model(self.components.feature_extractor)

        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        """准备训练参数"""
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        trainable_module_name = get_trainable_module_name(self.components)
        for attr_name, component in vars(self.components).items():
            if not hasattr(component, 'requires_grad_'):
                continue
            if attr_name == 'transformer':
                for layer_name, states_params in component.state_dict().items():
                    if layer_name in trainable_module_name:
                        states_params.requires_grad_(True)
                    else:
                        states_params.requires_grad_(False)
            else:
                component.requires_grad_(False)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()


    def train(self) -> None:
        """train"""
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
                self.args.batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,  # 需要训练的参数
            "total samples": len(self.dataset),  # len(dataset)
            "train epochs": self.args.train_epochs,  # epoch
            "train steps": self.args.train_steps,  # 总训练batch次数
            "batches per device": self.args.batch_size,  # batch_size
            "total batches observed per epoch": len(self.data_loader),  # len(data_loader)
            "train batch size total count": self.state.total_batch_size_count,  # 所有设备的batch_size
            "gradient accumulation steps": self.args.gradient_accumulation_steps,  # 梯度累计
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,  # ckpt_path
            initial_global_step,  # 初始ckpt已经训练的次数
            global_step,  # 现在已经训练的次数
            first_epoch,  # 初始已经训练的epoch数
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)
        # progress_bar: global_step / train_step
        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()

        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    # 在分布式训练中计算梯度范数、梯度裁剪
                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = (
                        self.args.do_validation and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

        accelerator.wait_for_everyone()
        # 训练结束，保存模型 + 验证 + 释放空间
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        """
        :param batch:
            {
            "prompt_embeddings"                 [[1,l,dim], [2,l,dim]]
            "current_prompt_embedding":         [b,l,dim]
            "image_embeddings":                 [[1,l,dim],[2,l,dim]]
            "current_image_embedding":          [b,l,dim]
            "current_video_latent":             [b,C,F,H,W]
            "current_image_feature":            [b,l,dim]

            "video_metadata": {
                "prompt_regress": prompt_regress,                   # [b,f]
                "video_path_regress": video_regress,                # [b,f]
                "num_frames": encoded_video.shape[1],               # [b,F]
                "height": encoded_video.shape[2],                   # [b,H]
                "width": encoded_video.shape[3],                    # [b,W]
            }
        :return:
        """
        # [b,l,dim]
        current_prompt_embedding = batch["current_prompt_embedding"]
        # [b,l,dim]
        current_image_embedding = batch["current_image_feature"]
        # [b,c,f,h,w]
        current_video_latent = batch["current_video_latent"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        # 填充第一帧，使得可以在时间维度分块
        if patch_size_t is not None:
            ncopy = current_video_latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = current_video_latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            current_video_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), current_video_latent], dim=2)
            assert current_video_latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = current_video_latent.shape

        # Get prompt embeddings
        _, seq_len, _ = current_prompt_embedding.shape
        current_prompt_embedding = current_prompt_embedding.view(batch_size, seq_len, -1).to(dtype=current_video_latent.dtype)
        _, seq_len, _ = current_image_embedding.shape
        current_image_embedding = current_image_embedding.view(batch_size, seq_len, -1).to(dtype=current_video_latent.dtype)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = current_video_latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds: RoPE位置编码（在pixel空间中计算，而不是latent空间）
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        # cos,sin: [temporal_size * grid_size_h * grid_size_w,  dim_t + dim_h + dim_w]
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,  # video h
                width=width * vae_scale_factor_spatial,  # video w
                num_frames=num_frames,  # latent f
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,  # 空间压缩率8
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        predicted_noise = self.components.transformer(
            hidden_states=latent_added_noise,  # [b,f,c,h,w]
            encoder_hidden_states=current_prompt_embedding,  # [b,L,dim]
            encoder_hidden_states_image=current_image_embedding,
            timestep=timesteps,  # [b,]
            image_rotary_emb=rotary_emb,
            # 位置编码:cos,sin:  [temporal_size * grid_size_h * grid_size_w, dim_t + dim_h + dim_w]
            return_dict=False,
        )[0]

        # v-prediction
        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_added_noise, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        """

        Args:
            prompt: str

        Returns:
            prompt_embedding: [1,L,dim]
        """
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def pred_image_feature(self, image_embeddings_batch, text_embeddings_batch, current_text_embeddings_batch):
        """
        image_embeddings: [[b,l,dim]]
        text_embeddings: [[b,l,dim]]
        current_text_embeddings: [b,l,dim]
        return: [b,l,dim]
        """
        feature_extractor = self.components.feature_extractor
        tmp_image_embeddings_batch, tmp_text_embedding_batch = [], []
        for image_embeddings, text_embeddings in zip(image_embeddings_batch, text_embeddings_batch):
            if len(image_embeddings) != 0:
                tmp_image_embeddings_batch.append(image_embeddings.to(device=self.accelerator.device))
            if len(text_embeddings) != 0:
                tmp_text_embedding_batch.append(text_embeddings.to(device=self.accelerator.device))

        # image_embeddings_batch = [image_embeddings.to(feature_extractor.device, dtype=feature_extractor.dtype) for
        #                           image_embeddings in image_embeddings_batch]
        # text_embeddings_batch = [text_embeddings.to(feature_extractor.device, dtype=feature_extractor.dtype) for
        #                          text_embeddings in text_embeddings_batch]
        current_text_embeddings_batch = current_text_embeddings_batch.to(self.accelerator.device)
        output = feature_extractor(tmp_image_embeddings_batch, tmp_text_embedding_batch, current_text_embeddings_batch)
        return output

    @override
    def encode_image(self, image_batch: list[Image]) -> torch.Tensor:
        # 预处理图像
        video_embeddings = encode_video(image_batch, self.components.image_encoder, num_frames=self.state.train_frames)
        return video_embeddings

    def prepare_rotary_positional_embeddings(
            self,
            height: int,  # video h
            width: int,  # video w
            num_frames: int,  # latent f
            transformer_config: Dict,  # transformer config
            vae_scale_factor_spatial: int,  # vae 空间缩放率
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return:
            cos, sin    shape都为: [temporal_size * grid_size_h * grid_size_w, dim_t + dim_h + dim_w]
            注意: grid_size_h，grid_size_w为video块数  temporal_size为latent块数
        """
        # 空间维度分块
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)
        # 时间维度分块: 块数向上取整
        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            # 块数向上取整
            base_num_frames = (
                                      num_frames + transformer_config.patch_size_t - 1
                              ) // transformer_config.patch_size_t
        # 3D RoPE
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


    # TODO: validate
    # def validate(self, batch):
    #     ...

    # TODO: pipline
    def initialize_pipeline(self):
        # pipline = CogVideoX_MultiPrompt_Pipeline()
        # TODO: 加载pipline
        pipline = None
        return pipline

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        return None

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(
                        self.components, name, component.to(self.accelerator.device, dtype=dtype)
                    )

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if (
                self.accelerator.distributed_type == DistributedType.DEEPSPEED
                or self.accelerator.is_main_process
        ):
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)


if __name__ == '__main__':
    args = Args.parse_args()
    trainer = CogVideoX_MultiTI2V_Trainer(args)
    trainer.fit()

"""
bash train_CogVideoX_ti2v.sh
"""

