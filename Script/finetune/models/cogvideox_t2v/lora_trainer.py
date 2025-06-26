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
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register


class CogVideoXT2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        """加载模型组件:vae,tokenizer,text_encoder,transformer,scheduler"""
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

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
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        return:
            {'prompt_embedding': [b,L,dim], 'encoded_videos': [b,c,f,h,w]}
        """
        ret = {"encoded_videos": [], "prompt_embedding": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        """根据{'prompt_embedding': [b,L,dim], 'encoded_videos': [b,c,f,h,w]}数据集计算loss"""
        # [b,L,dim]
        prompt_embedding = batch["prompt_embedding"]
        # [b,c,f,h,w]
        latent = batch["encoded_videos"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        # 填充第一帧，使得可以在时间维度分块
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds: RoPE位置编码（在pixel空间中计算，而不是latent空间）
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        # cos,sin: [temporal_size * grid_size_h * grid_size_w,  dim_t + dim_h + dim_w]
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,               # video h
                width=width * vae_scale_factor_spatial,                 # video w
                num_frames=num_frames,                                  # latent f
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,      # 空间压缩率8
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        predicted_noise = self.components.transformer(
            hidden_states=latent_added_noise,       # [b,f,c,h,w]
            encoder_hidden_states=prompt_embedding, # [b,L,dim]
            timestep=timesteps,                     # [b,]
            image_rotary_emb=rotary_emb,            # 位置编码:cos,sin:  [temporal_size * grid_size_h * grid_size_w, dim_t + dim_h + dim_w]
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
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,                    # video h
        width: int,                     # video w
        num_frames: int,                # latent f
        transformer_config: Dict,       # transformer config
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


register("cogvideox-t2v", "lora", CogVideoXT2VLoraTrainer)
