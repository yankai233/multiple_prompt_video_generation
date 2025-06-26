import sys
# sys.path.append(r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\Mutiple_prompt_mutiple_scene\Script\finetune\datasets')
sys.path.append('../../../')
sys.path.append('../../')


import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import os
import glob
import cv2
from PIL import Image

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from Mutiple_prompt_mutiple_scene.utils import load_json

from finetune.constants import LOG_LEVEL, LOG_NAME
import logging
try:
    from .utils import (
        load_prompts,
        load_videos,
        preprocess_video_with_buckets,
        preprocess_video_with_resize,
    )

    logger = get_logger(LOG_NAME, LOG_LEVEL)
except:
    from utils import (
        load_prompts,
        load_videos,
        preprocess_video_with_buckets,
        preprocess_video_with_resize,
    )
    from accelerate import Accelerator
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")
from Mutiple_prompt_mutiple_scene.utils import get_long_path

class BaseT2VDataset(Dataset):
    """
    Base dataset class for Text-to-Video (T2V) training.

    This dataset loads prompts and videos for T2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        root_path: str,             # root
        device: torch.device = None,
        trainer: "Trainer" = None,
        transform = None,
        max_regress_len = 10,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        root_path = str(root_path) if isinstance(root_path, Path) else root_path
        self.root_path = root_path
        logger.info('加载数据集中...')
        self.videos, self.prompts = self.get_video_prompt_pair(root_path)
        # 自回归片段: videos_regress: [[v1], [v1,v2]], prompts_regress:[[p1], [p1,p2]]
        self.prompts_regress = []
        self.videos_regress = []
        for i in range(len(self.videos)):
            for j in range(min(max_regress_len, len(self.videos[i]))):
                self.videos_regress.append(self.videos[i][:j + 1])
                self.prompts_regress.append(self.prompts[i][:j + 1])
        logger.info('加载数据集完成')

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])

        self.device = device
        # encode video
        self.encode_video = trainer.encode_video
        # encode text
        self.encode_text = trainer.encode_text
        # encode image
        self.encode_image = trainer.encode_image
        # 预测视频语义
        self.pred_image_feature = trainer.pred_image_feature
        self.trainer = trainer

        # Check if number of prompts matches number of videos
        if len(self.prompts_regress) != len(self.videos_regress):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts_regress)=} and {len(self.videos_regress)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

    def get_video_prompt_pair(self, root_path):
        """获取[[[v_1,p_1], [v_2,p_2]]]数据对"""
        prompt_dir = os.path.join(root_path, 'prompts')
        video_dir = os.path.join(root_path, 'videos')
        movie_prompt_lst = glob.glob(os.path.join(prompt_dir, '*.json'))

        total_video_lst = []
        total_prompt_lst = []
        for file_name in movie_prompt_lst:
            base_name = os.path.basename(file_name).split('.')[0]
            if os.path.exists(os.path.join(video_dir, base_name)):
                video_lst = []
                prompt_lst = []
                video_prompt_dict: dict = load_json(file_name)
                for video_path, prompt in video_prompt_dict.items():
                    video_path = os.path.join(video_dir, video_path)
                    if os.path.exists(video_path):
                        video_lst.append(video_path)
                        prompt_lst.append(prompt)
                total_video_lst.append(video_lst)
                total_prompt_lst.append(prompt_lst)
        return total_video_lst, total_prompt_lst

    def __len__(self) -> int:
        return len(self.videos_regress)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns:
            prompt_embedding                [f,l,dim]
            current_prompt_embedding        [l,dim]
            image_embedding                 [f,l2,dim2]
            current_image_embedding         [l2,dim2]
            current_video_latent            [F,c,h,w]
            prompt                          [f,]
            video_path                      [f,]
        """
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index
        prompt_embedding_lst = []
        image_embedding_lst = []

        prompt_regress = self.prompts_regress[index]
        video_regress = self.videos_regress[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        # cache: 将prompt embedding和encode video缓存，防止重复计算
        cache_dir = self.trainer.args.data_root / "cache" / "train_CogVideoX"
        video_latent_dir = (
            cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        )
        keyframe_embedding_dir = cache_dir / "video_embeddings" / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        current_image_feature_dir = cache_dir / "current_video_feature" / train_resolution_str

        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        keyframe_embedding_dir.mkdir(parents=True, exist_ok=True)
        current_image_feature_dir.mkdir(parents=True, exist_ok=True)

        # prompt_embedding
        for prompt in prompt_regress:
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            prompt_embedding_path = get_long_path(prompt_embedding_path)
            # 编码prompt
            need_encode_text = True
            if os.path.exists(prompt_embedding_path):  # prompt embedding已经存储,加载
                try:
                    prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
                    need_encode_text = False
                except:
                    need_encode_text = True
            if need_encode_text:  # embedding 并存储
                prompt_embedding = self.encode_text([prompt])[0]
                prompt_embedding = prompt_embedding.to("cpu")
                # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            prompt_embedding_lst.append(prompt_embedding)
        current_prompt_embedding = prompt_embedding_lst[-1]
        if len(prompt_embedding_lst) > 1:
            prompt_embedding_lst = torch.stack(prompt_embedding_lst[:-1], dim=0)
        else:
            prompt_embedding_lst = []

        # video_embedding
        for video_path in video_regress:
            video_hash_path = keyframe_embedding_dir / (str(hashlib.sha256(video_path.encode()).hexdigest()) + ".safetensors")
            video_hash_path = get_long_path(video_hash_path)

            need_encode_image = True
            if os.path.exists(video_hash_path):
                try:
                    image_embedding = load_file(video_hash_path)['video_embedding']
                    need_encode_image = False
                except:
                    need_encode_image = True
            if need_encode_image:
                image_embedding = self.encode_image([video_path])[0]
                image_embedding = image_embedding.to("cpu")
                save_file({"video_embedding": image_embedding}, video_hash_path)
            image_embedding_lst.append(image_embedding)
        current_image_embedding = image_embedding_lst[-1]
        if len(image_embedding_lst) > 1:
            image_embedding_lst = torch.stack(image_embedding_lst[:-1], dim=0)
        else:
            image_embedding_lst = []

        # current_video_latent
        current_video_path = video_regress[-1]
        encoded_video_path = video_latent_dir / (str(hashlib.sha256(current_video_path.encode()).hexdigest())+ ".safetensors")
        encoded_video_path = get_long_path(encoded_video_path)
        # 编码视频
        need_encode_video_latent = True
        if os.path.exists(encoded_video_path):
            # encoded_video = torch.load(encoded_video_path, weights_only=True)
            try:
                encoded_video = load_file(encoded_video_path)["encoded_video"]
                need_encode_video_latent = False
            except:
                need_encode_video_latent = True
            # shape of image: [C, H, W]
        if need_encode_video_latent:
            # 读取video,根据给定的[(f,h,w)]对视频裁剪
            frames = self.preprocess(current_video_path)
            frames = frames.to(self.device)
            # Current shape of frames: [F, C, H, W]， clamp(-1,1)
            frames = self.video_transform(frames)
            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            # vae编码
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)

        # current_image_feature
        current_image_feature_path = current_image_feature_dir / (str(hashlib.sha256(current_video_path.encode()).hexdigest()) + ".safetensors")
        current_image_feature_path = get_long_path(current_image_feature_path)
        need_predict_image_feature = True
        if os.path.exists(current_image_feature_path):
            try:
                current_image_feature = load_file(current_image_feature_path)["current_image_feature"]
                logger.debug(f"Loaded current image feature from {current_image_feature_path}", main_process_only=False)
                need_predict_image_feature = False
            except:
                need_predict_image_feature = True
        if need_predict_image_feature:
            current_image_feature = self.pred_image_feature([image_embedding_lst], [prompt_embedding_lst], current_prompt_embedding[None, :, :])
            current_image_feature = current_image_feature[0].to('cpu')
            save_file({'current_image_feature': current_image_feature}, current_image_feature_path)

        return {
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
            }
        }

    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class MultiTI2VDataset(BaseT2VDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        # clamp(-1,1)
        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        """读取视频,return: [max_num_frames,c, height, width]"""
        return preprocess_video_with_resize(
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """clamp(-1,1)"""
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)


###################################################################
#                        Test                                     #
###################################################################
class Logger:
    def info(self, s, **kwargs):
        print(s)

    def debug(self, s, **kwargs):
        print(s)

logger = Logger()

class BaseT2VDataset_Test(Dataset):
    """
    Base dataset class for Text-to-Video (T2V) training.

    This dataset loads prompts and videos for T2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
            self,
            root_path: str,  # root
            device: torch.device = None,

            transform=None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__()

        self.root_path = root_path
        logger.info('加载数据集中...')
        self.videos, self.prompts = self.get_video_prompt_pair(root_path)
        # 自回归片段: videos_regress: [[v1], [v1,v2]], prompts_regress:[[p1], [p1,p2]]
        self.prompts_regress = []
        self.videos_regress = []
        for i in range(len(self.videos)):
            for j in range(len(self.videos[i])):
                self.videos_regress.append(self.videos[i][:j + 1])
                self.prompts_regress.append(self.prompts[i][:j + 1])
        logger.info('加载数据集完成')

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])

        self.device = device
        # # encode video
        # self.encode_video = self.encode_video
        # # encode text
        # self.encode_text = .encode_text
        # # encode image
        # self.encode_image = trainer.encode_image
        # 预测关键帧语义
        # self.pred_image_feature = trainer.pred_image_feature
        # self.trainer = trainer

        # Check if number of prompts matches number of videos
        if len(self.prompts_regress) != len(self.videos_regress):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts_regress)=} and {len(self.videos_regress)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

    def encode_video(self, frames_batch: torch.Tensor) -> torch.Tensor:
        """
        :param frames:          [b,F,C,H,W]
        :return:
            [b,f,c,h,w]
        """
        return torch.randn([1, 13, 16, 32, 32])

    def encode_text(self, prompt_batch):
        """
        prompt_batch: [b,]
        :returns
            [b,l,dim]
        """
        return torch.randn([1, 77, 512])

    def encode_image(self, frames_batch: list[Image]) -> torch.Tensor:
        """
        :param frames:      [Image]
        :return:
            [b,l,dim]
        """
        return torch.randn([1, 77, 512])

    def pred_image_feature(self,image_embedding_lst_batch, prompt_embedding_lst_batch,
                                                            current_prompt_embedding_batch):
        """
        :param image_embedding_lst_batch:           [[1,l,dim], [2,l,dim]]
        :param prompt_embedding_lst_batch:          [[1,l,dim], [2,l,dim]]
        :param current_prompt_embedding_batch:      [b,l,dim]
        :return:
            [b,l,dim]
        """
        return torch.randn([1, 77, 512])

    def get_video_prompt_pair(self, root_path):
        """获取[[[v_1,p_1], [v_2,p_2]]]数据对"""
        prompt_dir = os.path.join(root_path, 'prompts')
        video_dir = os.path.join(root_path, 'videos')
        movie_prompt_lst = glob.glob(os.path.join(prompt_dir, '*.txt'))
        movie_video_lst = os.listdir(video_dir)

        total_video_lst = []
        total_prompt_lst = []
        for file_name in movie_prompt_lst:
            base_name = os.path.basename(file_name).split('.')[0]
            if os.path.exists(os.path.join(video_dir, base_name)):
                video_lst = []
                prompt_lst = []
                with open(file_name, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        video_path, prompt = line.split(',')
                        video_path = os.path.join(video_dir, video_path)
                        if os.path.exists(video_path):
                            video_lst.append(video_path)
                            prompt_lst.append(prompt)
                total_video_lst.append(video_lst)
                total_prompt_lst.append(prompt_lst)
        return total_video_lst, total_prompt_lst

    def get_Image_from_Clip(self, video_path, prompt):
        """根据视频片段和prompt提取该片段的某一帧作为锚点"""
        raise NotImplementedError("Subclass must implement this method")

    def __len__(self) -> int:
        return len(self.videos_regress)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns:
            prompt_embedding                [f,l,dim]
            current_prompt_embedding        [l,dim]
            image_embedding                 [f,l2,dim2]
            current_image_embedding         [l2,dim2]
            current_video_latent            [F,c,h,w]
            prompt                          [f,]
            video_path                      [f,]
        """
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index
        train_resolution = (49, 256, 256)
        data_root = Path(self.root_path)
        model_name = 'text_model'

        prompt_embedding_lst = []
        image_embedding_lst = []

        prompt_regress = self.prompts_regress[index]
        video_regress = self.videos_regress[index]

        train_resolution_str = "x".join(str(x) for x in train_resolution)

        # cache: 将prompt embedding和encode video缓存，防止重复计算
        cache_dir = data_root / "cache"
        video_latent_dir = (
                cache_dir / "video_latent" / model_name / train_resolution_str
        )
        keyframe_embedding_dir = cache_dir / "keyframe_embedding" / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        current_image_feature_dir = cache_dir / "current_image_feature" / train_resolution_str

        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        keyframe_embedding_dir.mkdir(parents=True, exist_ok=True)
        current_image_feature_dir.mkdir(parents=True, exist_ok=True)

        # prompt_embedding
        for prompt in prompt_regress:
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            # 编码prompt
            if prompt_embedding_path.exists():  # prompt embedding已经存储,加载
                prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
                # logger.debug(
                #     f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                #     main_process_only=False,
                # )
            else:  # embedding 并存储
                prompt_embedding = self.encode_text([prompt])
                prompt_embedding = prompt_embedding.to("cpu")
                # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
                prompt_embedding = prompt_embedding[0]
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
                logger.info(
                    f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False
                )
            prompt_embedding_lst.append(prompt_embedding)
        current_prompt_embedding = prompt_embedding_lst[-1]
        if len(prompt_embedding_lst) > 1:
            prompt_embedding_lst = torch.stack(prompt_embedding_lst[:-1], dim=0)
        else:
            prompt_embedding_lst = []

        # image_embedding
        for video_path in video_regress:

            image_hash = keyframe_embedding_dir / (str(hashlib.sha256(video_path.encode()).hexdigest()) + ".safetensors")
            if image_hash.exists():
                image_embedding = load_file(image_hash)['image_embedding']
            else:
                keyframe = self.get_Image_from_Clip(video_path, prompt_regress)
                image_embedding = self.encode_image([keyframe])[0]
                image_embedding = image_embedding.to("cpu")
                save_file({"image_embedding": image_embedding}, image_hash)
                logger.info(f'Saved keyframe embedding to {image_hash}', main_process_only=False)
            image_embedding_lst.append(image_embedding)
        current_image_embedding = image_embedding_lst[-1]
        if len(image_embedding_lst) > 1:
            image_embedding_lst = torch.stack(image_embedding_lst[:-1], dim=0)
        else:
            image_embedding_lst = []

        # current_video_latent
        current_video_path = video_regress[-1]
        encoded_video_path = video_latent_dir / (str(hashlib.sha256(current_video_path.encode()).hexdigest()) + ".safetensors")
        # 编码视频
        if encoded_video_path.exists():
            # encoded_video = torch.load(encoded_video_path, weights_only=True)
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            # shape of image: [C, H, W]
        else:
            # 读取video,根据给定的[(f,h,w)]对视频裁剪
            frames = self.preprocess(current_video_path)
            frames = frames.to(self.device)
            # Current shape of frames: [F, C, H, W]， clamp(-1,1)
            frames = self.video_transform(frames)
            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            # vae编码
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        # current_image_feature
        current_image_feature_path = current_image_feature_dir / (str(hashlib.sha256(current_video_path.encode()).hexdigest()) + ".safetensors")
        if current_image_feature_path.exists():
            current_image_feature = load_file(current_image_feature_path)["current_image_feature"]
            logger.debug(f"Loaded current image feature from {current_image_feature_path}", main_process_only=False)
        else:
            current_image_feature = self.pred_image_feature([image_embedding_lst], [prompt_embedding_lst],
                                                            current_prompt_embedding[None,:,:])[0]
            save_file({'current_image_feature': current_image_feature}, current_image_feature_path)
            logger.info(f'Saved current_image_feature to {current_image_feature_path}', main_process_only=False)

        return {
            "prompt_embeddings": prompt_embedding_lst,  # [f-1,l,dim]
            "current_prompt_embedding": current_prompt_embedding,  # [l,dim]
            "image_embeddings": image_embedding_lst,  # [f-1,l,dim]
            "current_image_embedding": current_image_embedding,  # [l,dim]
            "current_video_latent": encoded_video,  # [C,F,H,W]
            "current_image_feature": current_image_feature,  # [l,dim]

            "video_metadata": {
                "prompt_regress": prompt_regress,  # [f,]
                "video_path_regress": video_regress,  # [f,]
                "num_frames": encoded_video.shape[1],  # F
                "height": encoded_video.shape[2],  # H
                "width": encoded_video.shape[3],  # W
            },
        }

    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class MultiTI2VDataset_Test(BaseT2VDataset_Test):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        # clamp(-1,1)
        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )

    @override
    def get_Image_from_Clip(self, video_path, prompt):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(rgb_frame)
            return frame_pil
        else:
            raise Exception(f'读取{video_path}失败')

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        """读取视频,return: [max_num_frames,c, height, width]"""
        return preprocess_video_with_resize(
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """clamp(-1,1)"""
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)


class T2VDatasetWithBuckets(BaseT2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],       # 分辨率:[(frame,h,w),...]
        vae_temporal_compression_ratio: int,                        # vae 时间压缩率
        vae_height_compression_ratio: int,                          # vae 空间压缩率
        vae_width_compression_ratio: int,                           # vae 空间压缩率
        *args,
        **kwargs,
    ) -> None:
        """ """
        super().__init__(*args, **kwargs)
        # vae压缩后分辨率: [(frame/4,h/8,w/8)]
        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        # clamp(-1,1)
        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        """读取mp4"""
        return preprocess_video_with_buckets(video_path, self.video_resolution_buckets)

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)


if __name__ == '__main__':
    root_path = r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\excluded_dir\dataset'
    device = 'cpu'
    dataset = MultiTI2VDataset_Test(max_num_frames=144, height=256, width=256, device=device, root_path=root_path)
    for i in range(len(dataset)):
        for keys, value in dataset[i].items():
            print(keys, '.shape:', value.shape if isinstance(value, torch.Tensor) else len(value))
