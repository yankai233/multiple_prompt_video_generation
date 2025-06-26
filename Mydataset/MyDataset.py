import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../../'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from PIL import Image
import random
from Mutiple_prompt_mutiple_scene.utils import *
import logging

import glob
import cv2
import copy
from pathlib import Path
import decord
import hashlib
from safetensors.torch import load_file, save_file
from Mutiple_prompt_mutiple_scene.utils import *
from Mutiple_prompt_mutiple_scene.Script.APIServer.Gemini_API import Gemini_API, response_func, get_gemini_prompt_from_past_and_current, get_token_index_lst, SYSTEM_MESSAGE
from Mutiple_prompt_mutiple_scene.Mydataset.data_utils import *


class Feature_Extraction_Dataset(data.Dataset):
    def __init__(
            self,
            root_path,
            height,
            width,
            cache_dir_name='',
            logger=None,
            video_encoder=None,
            video_processor=None,
            text_encoder=None,
            tokenizer=None,
            gemini_api: Optional[Gemini_API] = None,
            device=None,
            transform=None,

            use_feature_extractor=False,
            max_regress_len=10,
            api_start_prompt=None,

            num_frames=49,

    ):
        super().__init__()
        self.logger = logger
        if logger is None:
            self.init_logging()

        # 加载数据集[[[V_1_path,P_1], [V_2_path,P_2]], [[V_1_path,P_1],...]]
        self.root_path = root_path
        self.height = height
        self.width = width
        self.cache_dir_name = cache_dir_name

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

        self.device = device if device == 'cpu' or 'cuda' in device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # encode text
        self.text_encoder = text_encoder.to(self.device)
        self.tokenizer = tokenizer
    
        self.video_encoder = video_encoder.to(self.device)
        self.video_processor = video_processor
        # Gemini API
        self.use_feature_extractor = use_feature_extractor
        if use_feature_extractor:
            self.Gemini_API = gemini_api
            self.api_start_prompt = api_start_prompt
        else:
            self.Gemini_API = None
        # Check if number of prompts matches number of videos
        if len(self.prompts_regress) != len(self.videos_regress):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts_regress)=} and {len(self.videos_regress)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        self.num_frames = num_frames

    def __getitem__(self, index):
        """
        :returns
        {
            'video_embeddings': video_embedding_lst,            # [[f,l,dim],..]
            'prompt_embeddings': prompt_embedding_lst,          # [[f,l,dim],..]
            'current_video_embedding': current_video_embedding,     # [b,l,dim]
            'current_prompt_embedding': current_prompt_embedding    # [b,l,dim]
            'meta_info':
                {
                 'past_prompt_lst': past_prompt_lst,            # [['b11','b12']]
                 'current_prompt': current_prompt,              # ['b1','b2']
                 'filter_past_prompt_lst': filter_prompt_lst[:-1],  # [['b11','b12']]
                 'filter_current_prompt': filter_prompt_lst[-1],    # ['b1', 'b2']
                 'filter_past_token_index': token_index_lst[:-1]            # [[[batch1_scene1_token1,...],...],...]
                 'filter_current_token_index': token_index_lst[-1]          # [[batch1_token1,...],...]
                }
        }
        5. TODO: 使用预训练的transformer模型
        """
        logger = self.logger
        error_num = 3
        while error_num > 0:
            try:
                # data: [[v_1,p_1], [v_2,p_2],...]
                video_regress, prompts_regress = self.videos_regress[index], self.prompts_regress[index]
                prompts_regress = copy.deepcopy(prompts_regress)
                break
            except Exception as e:
                error_num -= 1
                index = random.randint(0, len(self.videos_regress) - 1)
                self.logger.warning(e)
        assert error_num > 0, '加载数据集错误'
        # 编码
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
        video_embedding_lst = []

        train_resolution_str = f'{self.height}x{self.width}'
        data_root = Path(self.root_path)
        # cache: 将prompt embedding和encode video缓存，防止重复计算
        cache_dir = data_root / "cache" / "train_Feature_Extraction" / self.cache_dir_name

        video_embedding_dir = cache_dir / "video_embeddings" / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"

        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        video_embedding_dir.mkdir(parents=True, exist_ok=True)

        current_prompt, past_prompt_lst = prompts_regress[-1], prompts_regress[:-1]
        if self.use_feature_extractor and self.Gemini_API is not None:
            filter_prompt_dir = cache_dir / "filter_prompt" / self.Gemini_API.model_name

            filter_prompt_dir.mkdir(parents=True, exist_ok=True)
            # 1. token筛选
            video_dir_name = path_join(video_regress[-1]).split('/')[-3]
            filter_prompt_path = filter_prompt_dir / f'{video_dir_name}.json'
            filter_prompt_path = get_path_str(filter_prompt_path)

            response_dict = {}
            if os.path.exists(filter_prompt_path):
                response_dict = load_json(filter_prompt_path)


            key = get_gemini_prompt_from_past_and_current(current_prompt, past_prompt_lst)
            if response_dict is None or key not in response_dict:
                response_dict = self.Gemini_API.chat_feature_filter(current_prompt=current_prompt, start_prompt=self.api_start_prompt, past_prompt_lst=past_prompt_lst,
                                                response_func=response_func, response_dict=response_dict, save_path=filter_prompt_path)
            # 获取token index
            output_dict = response_dict[key]
            filter_prompt_lst = list(output_dict.values())
            token_index_lst = get_token_index_lst(filter_prompt_lst[:len(past_prompt_lst)] + [filter_prompt_lst[-1]], past_prompt_lst, current_prompt)

        # 2. prompt_embedding
        for prompt in prompts_regress:
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            prompt_embedding_path = get_long_path(prompt_embedding_path)
            # 编码prompt
            need_encode_text = True
            if os.path.exists(prompt_embedding_path):  # prompt embedding已经存储,加载
                try:
                    prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
                    need_encode_text = False
                except Exception as e:
                    logger.error(f'!!! error loading: {prompt_embedding_path}')
                    need_encode_text = True
            if need_encode_text:  # embedding 并存储
                # [seq_len, hidden_size]
                prompt_embedding = encode_text([prompt], self.tokenizer, self.text_encoder)[0]
                prompt_embedding = prompt_embedding.to("cpu")
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)

            prompt_embedding_lst.append(prompt_embedding)
        current_prompt_embedding = prompt_embedding_lst[-1]

        if len(prompt_embedding_lst) > 1:
            prompt_embedding_lst = torch.stack(prompt_embedding_lst[:-1], dim=0)
        else:
            prompt_embedding_lst = []

        # 3. video embedding
        for i in range(len(video_regress)):
            video_path = video_regress[i]
            video_hash_path = video_embedding_dir / (
                        str(hashlib.sha256(video_path.encode()).hexdigest()) + ".safetensors")
            video_hash_path = get_long_path(video_hash_path)
            need_encode_video = True
            if os.path.exists(video_hash_path):
                try:
                    video_embedding = load_file(video_hash_path)['video_embedding']
                    need_encode_video = False
                except Exception as e:
                    print(f'!!! error loading: {video_hash_path}')
                    need_encode_video = True
            if need_encode_video:
                video_embedding = encode_video([video_path], self.video_encoder, video_size=224, num_frames=self.num_frames)[0]
                video_embedding = video_embedding.to("cpu")
                save_file({"video_embedding": video_embedding}, video_hash_path)

            video_embedding_lst.append(video_embedding)
        current_video_embedding = video_embedding_lst[-1]
        if len(video_embedding_lst) > 1:
            video_embedding_lst = torch.stack(video_embedding_lst[:-1], dim=0)
        else:
            video_embedding_lst = []

        # return
        data_dict = {
            'video_embeddings': video_embedding_lst, 'prompt_embeddings': prompt_embedding_lst,
            'current_video_embedding': current_video_embedding,
            'current_prompt_embedding': current_prompt_embedding
        }
        if self.use_feature_extractor and self.Gemini_API is not None:
            data_dict['meta_info'] = {
                        'past_prompt_lst': past_prompt_lst, 'current_prompt': current_prompt,
                        'filter_past_prompt_lst': filter_prompt_lst[:-1],
                        'filter_current_prompt': filter_prompt_lst[-1],
                        'filter_past_token_index': token_index_lst[:-1],
                        'filter_current_token_index': token_index_lst[-1]
            }
        else:
            data_dict['meta_info'] = {
                'past_prompt_lst': past_prompt_lst, 'current_prompt': current_prompt
            }

        return data_dict

    def __len__(self):
        return len(self.videos_regress)

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
                video_prompt_dict:dict = load_json(file_name)
                for video_path, prompt in video_prompt_dict.items():
                    video_path = os.path.join(video_dir, video_path)
                    if os.path.exists(video_path):
                        video_lst.append(video_path)
                        prompt_lst.append(prompt)
                total_video_lst.append(video_lst)
                total_prompt_lst.append(prompt_lst)
        return total_video_lst, total_prompt_lst


    def init_logging(self):
        # 配置日志
        import logging
        import colorlog
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            handlers=[
                # 把日志同时输出到文件和控制台
                logging.FileHandler('../excluded_dir/output_dir/logs/1/train_fe_log.txt', mode='w', encoding='utf-8'),
            ]
        )

        # 创建一个控制台处理器，并使用 colorlog 格式化
        console_handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(formatter)

        # 获取日志记录器
        logger = logging.getLogger(__name__)
        logger.addHandler(console_handler)
        self.logger = logger


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    samples:
    [{
        "prompt_embeddings": prompt_embedding_lst,              # [f-1,l,dim]
        "current_prompt_embedding": current_prompt_embedding,   # [l,dim]
        "video_embeddings": video_embedding_lst,                # [f-1,l,dim]
        "current_video_embedding": current_video_embedding,     # [l,dim]
        'meta_info':
            {
             'past_prompt_lst': past_prompt_lst,                # ['1','2']
             'current_prompt': current_prompt,                  # '1'
             'filter_past_prompt_lst': filter_prompt_lst[:-1],  # ['1','2']
             'filter_current_prompt': filter_prompt_lst[-1],    # '1'
             'filter_past_token_index': token_index_lst[:-1],   # [[1,2],[1,2]]
             'filter_current_token_index': token_index_lst[-1]  # [1,2]
            }
    },...]
    """
    res = {'prompt_embeddings': [], 'current_prompt_embedding': [], 'video_embeddings': [],
     'current_video_embedding': [],
     'meta_info': {
         'past_prompt_lst': [], 'current_prompt': []
        }
     }
    use_feature_extractor = 'filter_past_prompt_lst' in samples[0]['meta_info']
    if use_feature_extractor:
        res['meta_info'] = {
            'past_prompt_lst': [], 'current_prompt': [],
            'filter_past_prompt_lst': [], 'filter_current_prompt': [],
            'filter_past_token_index': [], 'filter_current_token_index': []
        }

    for sample in samples:
        res['prompt_embeddings'].append(sample['prompt_embeddings'])
        res['video_embeddings'].append(sample['video_embeddings'])

        res['current_prompt_embedding'].append(sample['current_prompt_embedding'])
        res['current_video_embedding'].append(sample['current_video_embedding'])

        res['meta_info']['past_prompt_lst'].append(sample['meta_info']['past_prompt_lst'])
        res['meta_info']['current_prompt'].append(sample['meta_info']['current_prompt'])
        if use_feature_extractor:
            res['meta_info']['filter_past_prompt_lst'].append(sample['meta_info']['filter_past_prompt_lst'])
            res['meta_info']['filter_current_prompt'].append(sample['meta_info']['filter_current_prompt'])
            res['meta_info']['filter_past_token_index'].append(sample['meta_info']['filter_current_prompt'])
            res['meta_info']['filter_current_token_index'].append(sample['meta_info']['filter_current_token_index'])

    # stack
    res['current_prompt_embedding'] = torch.stack(res['current_prompt_embedding'], dim=0)
    res['current_video_embedding'] = torch.stack(res['current_video_embedding'], dim=0)

    return res



# class Feature_Extraction_Dataset_video_Encoder(data.Dataset):
#     def __init__(
#             self,
#             root_path,
#             height,
#             width,
#             cache_dir_name='',
#             logger=None,
#             video_encoder=None,
#             video_processor=None,
#             text_encoder=None,
#             tokenizer=None,
#             gemini_api: Optional[Gemini_API] = None,
#             device=None,
#             transform=None,
#
#             use_feature_extractor=False,
#             max_regress_len=10,
#             api_start_prompt=None,
#
#     ):
#         super().__init__()
#         self.logger = logger
#         if logger is None:
#             self.init_logging()
#
#         # 加载数据集[[[V_1_path,P_1], [V_2_path,P_2]], [[V_1_path,P_1],...]]
#         self.root_path = root_path
#         self.height = height
#         self.width = width
#         self.cache_dir_name = cache_dir_name
#
#         logger.info('加载数据集中...')
#         self.videos, self.prompts = self.get_video_prompt_pair(root_path)
#         # 自回归片段: videos_regress: [[v1], [v1,v2]], prompts_regress:[[p1], [p1,p2]]
#         self.prompts_regress = []
#         self.videos_regress = []
#         for i in range(len(self.videos)):
#             for j in range(min(max_regress_len, len(self.videos[i]))):
#                 self.videos_regress.append(self.videos[i][:j + 1])
#                 self.prompts_regress.append(self.prompts[i][:j + 1])
#         logger.info('加载数据集完成')
#
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
#
#         self.device = device if device == 'cpu' or 'cuda' in device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         # encode text
#         self.text_encoder = text_encoder.to(self.device)
#         self.tokenizer = tokenizer
#         # TODO: encode video -> encode video
#         self.video_encoder = video_encoder.to(self.device)
#         self.processor = video_processor
#         # Gemini API
#         self.use_feature_extractor = use_feature_extractor
#         if use_feature_extractor:
#             self.Gemini_API = gemini_api
#             self.api_start_prompt = api_start_prompt
#         else:
#             self.Gemini_API = None
#         # Check if number of prompts matches number of videos
#         if len(self.prompts_regress) != len(self.videos_regress):
#             raise ValueError(
#                 f"Expected length of prompts and videos to be the same but found {len(self.prompts_regress)=} and {len(self.videos_regress)=}. Please ensure that the number of caption prompts and videos match in your dataset."
#             )
#
#     def __getitem__(self, index):
#         """
#         :returns
#         {
#             'video_embeddings': video_embedding_lst,            # [[f,l,dim],..]
#             'prompt_embeddings': prompt_embedding_lst,          # [[f,l,dim],..]
#             'current_video_embedding': current_video_embedding,     # [b,l,dim]
#             'current_prompt_embedding': current_prompt_embedding    # [b,l,dim]
#             'meta_info':
#                 {
#                  'past_prompt_lst': past_prompt_lst,            # [['b11','b12']]
#                  'current_prompt': current_prompt,              # ['b1','b2']
#                  'filter_past_prompt_lst': filter_prompt_lst[:-1],  # [['b11','b12']]
#                  'filter_current_prompt': filter_prompt_lst[-1],    # ['b1', 'b2']
#                  'filter_past_token_index': token_index_lst[:-1]            # [[[batch1_scene1_token1,...],...],...]
#                  'filter_current_token_index': token_index_lst[-1]          # [[batch1_token1,...],...]
#                 }
#         }
#         1. TODO: 加载的是视频而不是第一帧
#         2. TODO: 对图像编码变为对视频编码
#         3. TODO: 修改Feature_Extraction_Moudle_T5.yaml的参数
#
#         5. TODO: 使用预训练的transformer模型
#         """
#         logger = self.logger
#         error_num = 3
#         while error_num > 0:
#             try:
#                 # data: [[v_1,p_1], [v_2,p_2],...]
#                 video_regress, prompts_regress = self.videos_regress[index], self.prompts_regress[index]
#                 prompts_regress = copy.deepcopy(prompts_regress)
#                 video_lst, text_lst = [], []
#                 for i in range(len(video_regress)):
#                     frame = self.get_video_from_Clip(video_regress[i], prompts_regress)
#                     video_lst.append(frame)
#                 break
#             except Exception as e:
#                 error_num -= 1
#                 index = random.randint(0, len(self.videos_regress) - 1)
#                 self.logger.warning(e)
#         assert error_num > 0, '加载数据集错误'
#         # 编码
#         if isinstance(index, list):
#             # Here, index is actually a list of data objects that we need to return.
#             # The BucketSampler should ideally return indices. But, in the sampler, we'd like
#             # to have information about num_frames, height and width. Since this is not stored
#             # as metadata, we need to read the video to get this information. You could read this
#             # information without loading the full video in memory, but we do it anyway. In order
#             # to not load the video twice (once to get the metadata, and once to return the loaded video
#             # based on sampled indices), we cache it in the BucketSampler. When the sampler is
#             # to yield, we yield the cache data instead of indices. So, this special check ensures
#             # that data is not loaded a second time. PRs are welcome for improvements.
#             return index
#
#         prompt_embedding_lst = []
#         video_embedding_lst = []
#
#         train_resolution_str = f'{self.height}x{self.width}'
#         data_root = Path(self.root_path)
#         # cache: 将prompt embedding和encode video缓存，防止重复计算
#         cache_dir = data_root / "cache" / "train_Feature_Extraction" / self.cache_dir_name
#
#         keyframe_embedding_dir = cache_dir / "keyframe_embedding" / train_resolution_str
#         prompt_embeddings_dir = cache_dir / "prompt_embeddings"
#
#         prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
#         keyframe_embedding_dir.mkdir(parents=True, exist_ok=True)
#
#         current_prompt, past_prompt_lst = prompts_regress[-1], prompts_regress[:-1]
#         if self.use_feature_extractor and self.Gemini_API is not None:
#             filter_prompt_dir = cache_dir / "filter_prompt" / self.Gemini_API.model_name
#
#             filter_prompt_dir.mkdir(parents=True, exist_ok=True)
#             # 1. token筛选
#             video_dir_name = path_join(video_regress[-1]).split('/')[-3]
#             filter_prompt_path = filter_prompt_dir / f'{video_dir_name}.json'
#             filter_prompt_path = get_path_str(filter_prompt_path)
#
#             response_dict = {}
#             if os.path.exists(filter_prompt_path):
#                 response_dict = load_json(filter_prompt_path)
#
#
#             key = get_gemini_prompt_from_past_and_current(current_prompt, past_prompt_lst)
#             if response_dict is None or key not in response_dict:
#                 response_dict = self.Gemini_API.chat_feature_filter(current_prompt=current_prompt, start_prompt=self.api_start_prompt, past_prompt_lst=past_prompt_lst,
#                                                 response_func=response_func, response_dict=response_dict, save_path=filter_prompt_path)
#             # 获取token index
#             output_dict = response_dict[key]
#             filter_prompt_lst = list(output_dict.values())
#             token_index_lst = get_token_index_lst(filter_prompt_lst[:len(past_prompt_lst)] + [filter_prompt_lst[-1]], past_prompt_lst, current_prompt)
#
#         # 2. prompt_embedding
#         for prompt in prompts_regress:
#             prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
#             prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
#             prompt_embedding_path = get_long_path(prompt_embedding_path)
#             # 编码prompt
#             if os.path.exists(prompt_embedding_path):  # prompt embedding已经存储,加载
#                 prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
#                 logger.debug(
#                     f"Loaded prompt embedding from {prompt_embedding_path}"
#                 )
#             else:  # embedding 并存储
#                 prompt_embedding = self.encode_text([prompt])
#                 prompt_embedding = prompt_embedding.to("cpu")
#                 # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
#                 prompt_embedding = prompt_embedding[0]
#                 save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
#                 logger.info(
#                     f"Saved prompt embedding to {prompt_embedding_path}"
#                 )
#             prompt_embedding_lst.append(prompt_embedding)
#         current_prompt_embedding = prompt_embedding_lst[-1]
#         if len(prompt_embedding_lst) > 1:
#             prompt_embedding_lst = torch.stack(prompt_embedding_lst[:-1], dim=0)
#         else:
#             prompt_embedding_lst = []
#
#         # 3. video_embedding
#         for i in range(len(video_regress)):
#             video_path, keyframe = video_regress[i], video_lst[i]
#             video_hash = keyframe_embedding_dir / (
#                         str(hashlib.sha256(video_path.encode()).hexdigest()) + ".safetensors")
#             video_hash = get_long_path(video_hash)
#             if os.path.exists(video_hash):
#                 video_embedding = load_file(video_hash)['video_embedding']
#             else:
#                 video_embedding = self.encode_video([keyframe])[0]
#                 video_embedding = video_embedding.to("cpu")
#                 save_file({"video_embedding": video_embedding}, video_hash)
#                 logger.info(f'Saved keyframe embedding to {video_hash}')
#             video_embedding_lst.append(video_embedding)
#         current_video_embedding = video_embedding_lst[-1]
#         if len(video_embedding_lst) > 1:
#             video_embedding_lst = torch.stack(video_embedding_lst[:-1], dim=0)
#         else:
#             video_embedding_lst = []
#
#         # return
#         data_dict = {
#             'video_embeddings': video_embedding_lst, 'prompt_embeddings': prompt_embedding_lst,
#             'current_video_embedding': current_video_embedding,
#             'current_prompt_embedding': current_prompt_embedding
#         }
#         if self.use_feature_extractor and self.Gemini_API is not None:
#             data_dict['meta_info'] = {
#                         'past_prompt_lst': past_prompt_lst, 'current_prompt': current_prompt,
#                         'filter_past_prompt_lst': filter_prompt_lst[:-1],
#                         'filter_current_prompt': filter_prompt_lst[-1],
#                         'filter_past_token_index': token_index_lst[:-1],
#                         'filter_current_token_index': token_index_lst[-1]
#             }
#         else:
#             data_dict['meta_info'] = {
#                 'past_prompt_lst': past_prompt_lst, 'current_prompt': current_prompt
#             }
#
#         return data_dict
#
#     def __len__(self):
#         return len(self.videos_regress)
#
#     def get_video_prompt_pair(self, root_path):
#         """获取[[[v_1,p_1], [v_2,p_2]]]数据对"""
#         prompt_dir = os.path.join(root_path, 'prompts')
#         video_dir = os.path.join(root_path, 'videos')
#         movie_prompt_lst = glob.glob(os.path.join(prompt_dir, '*.json'))
#
#         total_video_lst = []
#         total_prompt_lst = []
#         for file_name in movie_prompt_lst:
#             base_name = os.path.basename(file_name).split('.')[0]
#             if os.path.exists(os.path.join(video_dir, base_name)):
#                 video_lst = []
#                 prompt_lst = []
#                 video_prompt_dict:dict = load_json(file_name)
#                 for video_path, prompt in video_prompt_dict.items():
#                     video_path = os.path.join(video_dir, video_path)
#                     if os.path.exists(video_path):
#                         video_lst.append(video_path)
#                         prompt_lst.append(prompt)
#                 total_video_lst.append(video_lst)
#                 total_prompt_lst.append(prompt_lst)
#         return total_video_lst, total_prompt_lst
#
#     # TODO:
#     def get_video_from_Clip(self, video_path, prompts):
#         """根据视频片段和prompt提取该片段的某一帧作为锚点"""
#         if isinstance(video_path, str):
#             video_path = Path(video_path)
#
#         try:
#             # 打开视频文件
#             cap = cv2.VideoCapture(str(video_path))
#
#             # 检查视频是否成功打开
#             if not cap.isOpened():
#                 self.logger.error(f"无法打开视频文件: {video_path}")
#                 return None
#
#             # 读取第一帧
#             ret, frame = cap.read()
#
#             # 检查是否成功读取帧
#             if ret:
#                 # 调整帧的大小
#                 frame = cv2.resize(frame, (self.width, self.height))
#
#                 # 将 BGR 格式转换为 RGB 格式（OpenCV 默认使用 BGR 格式，而 PIL 使用 RGB 格式）
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#                 # 将 OpenCV 图像转换为 PIL 图像
#                 pil_video = video.fromarray(frame)
#
#                 return pil_video
#             else:
#                 print("无法读取视频帧")
#                 return None
#
#         except Exception as e:
#             self.logger.error(f"发生错误: {e}")
#             return None
#         finally:
#             # 释放视频捕获对象
#             if 'cap' in locals() and cap.isOpened():
#                 cap.release()
#
#     def encode_text(self, prompt):
#         """
#         将prompt编码为embedding
#         :param prompt:   str, [str]
#         :return:
#             [b,77,dim]
#         """
#         if isinstance(prompt, str):
#             prompt = [prompt]
#         text_input = self.tokenizer(
#             prompt,
#             padding="max_length",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             add_special_tokens=True,
#             return_tensors="pt",
#         )
#         text_embeddings = self.text_encoder(
#             text_input.input_ids.to(self.text_encoder.device)
#         )[0]
#         return text_embeddings
#
#     # TODO:
#     def encode_video(self, video_batch: list[video]) -> torch.Tensor:
#         # 预处理图像
#         inputs = self.processor(videos=video_batch, return_tensors="pt", padding=True, truncation=True)
#         inputs = inputs.to(self.video_encoder.device)
#
#         # 编码图像
#         with torch.no_grad():
#             video_features = self.video_encoder(**inputs)[0]  # 形状为 [b, seq_len, embedding_dim]
#         return video_features
#
#     def init_logging(self):
#         # 配置日志
#         import logging
#         import colorlog
#         # 配置日志
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s %(message)s',
#             handlers=[
#                 # 把日志同时输出到文件和控制台
#                 logging.FileHandler('../excluded_dir/output_dir/logs/1/train_fe_log.txt', mode='w', encoding='utf-8'),
#             ]
#         )
#
#         # 创建一个控制台处理器，并使用 colorlog 格式化
#         console_handler = logging.StreamHandler()
#         formatter = colorlog.ColoredFormatter(
#             '%(log_color)s%(asctime)s %(message)s',
#             log_colors={
#                 'DEBUG': 'cyan',
#                 'INFO': 'green',
#                 'WARNING': 'yellow',
#                 'ERROR': 'red',
#                 'CRITICAL': 'red,bg_white',
#             }
#         )
#         console_handler.setFormatter(formatter)
#
#         # 获取日志记录器
#         logger = logging.getLogger(__name__)
#         logger.addHandler(console_handler)
#         self.logger = logger
#
#
# def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     samples:
#     [{
#         "prompt_embeddings": prompt_embedding_lst,              # [f-1,l,dim]
#         "current_prompt_embedding": current_prompt_embedding,   # [l,dim]
#         "video_embeddings": video_embedding_lst,                # [f-1,l,dim]
#         "current_video_embedding": current_video_embedding,     # [l,dim]
#         'meta_info':
#             {
#              'past_prompt_lst': past_prompt_lst,                # ['1','2']
#              'current_prompt': current_prompt,                  # '1'
#              'filter_past_prompt_lst': filter_prompt_lst[:-1],  # ['1','2']
#              'filter_current_prompt': filter_prompt_lst[-1],    # '1'
#              'filter_past_token_index': token_index_lst[:-1],   # [[1,2],[1,2]]
#              'filter_current_token_index': token_index_lst[-1]  # [1,2]
#             }
#     },...]
#     """
#     res = {'prompt_embeddings': [], 'current_prompt_embedding': [], 'video_embeddings': [],
#      'current_video_embedding': [],
#      'meta_info': {
#          'past_prompt_lst': [], 'current_prompt': []
#         }
#      }
#     use_feature_extractor = 'filter_past_prompt_lst' in samples[0]['meta_info']
#     if use_feature_extractor:
#         res['meta_info'] = {
#             'past_prompt_lst': [], 'current_prompt': [],
#             'filter_past_prompt_lst': [], 'filter_current_prompt': [],
#             'filter_past_token_index': [], 'filter_current_token_index': []
#         }
#
#     for sample in samples:
#         res['prompt_embeddings'].append(sample['prompt_embeddings'])
#         res['video_embeddings'].append(sample['video_embeddings'])
#
#         res['current_prompt_embedding'].append(sample['current_prompt_embedding'])
#         res['current_video_embedding'].append(sample['current_video_embedding'])
#
#         res['meta_info']['past_prompt_lst'].append(sample['meta_info']['past_prompt_lst'])
#         res['meta_info']['current_prompt'].append(sample['meta_info']['current_prompt'])
#         if use_feature_extractor:
#             res['meta_info']['filter_past_prompt_lst'].append(sample['meta_info']['filter_past_prompt_lst'])
#             res['meta_info']['filter_current_prompt'].append(sample['meta_info']['filter_current_prompt'])
#             res['meta_info']['filter_past_token_index'].append(sample['meta_info']['filter_current_prompt'])
#             res['meta_info']['filter_current_token_index'].append(sample['meta_info']['filter_current_token_index'])
#
#     # stack
#     res['current_prompt_embedding'] = torch.stack(res['current_prompt_embedding'], dim=0)
#     res['current_video_embedding'] = torch.stack(res['current_video_embedding'], dim=0)
#
#     return res


if __name__ == '__main__':
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPVisionModel
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        # handlers=[
        #     # 把日志同时输出到文件和控制台
        #     logging.FileHandler(os.path.abspath('./excluded_dir/output_dir/logs/1/dataset_log.txt'), mode='w', encoding='utf-8'),
        #     logging.StreamHandler()
        # ]
    )
    logger = logging.getLogger(__name__)
    model_id = r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\excluded_dir\local_model\model_dir\Clip4Clip'
    text_encoder = CLIPTextModel.from_pretrained(model_id, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
    video_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, local_files_only=True)

    height, width = 256, 256
    dataset = Feature_Extraction_Dataset(r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\excluded_dir\dataset',
        height, width, '', logger, video_encoder, text_encoder, tokenizer,  device='cpu')

    dataloader = data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        # videos: [[1,3,h,w],[2,3,h,w]]
        prompt_embeddings = batch['prompt_embeddings']
        video_embeddings = batch['video_embeddings']
        current_prompt_embedding = batch['current_prompt_embedding']
        current_video_embedding = batch['current_video_embedding']

        # filter_current_prompt = batch['meta_info']['filter_current_prompt']
        # filter_past_prompt_lst = batch['meta_info']['filter_past_prompt_lst']
        if isinstance(prompt_embeddings[1], torch.Tensor):
            print(len(prompt_embeddings), prompt_embeddings[1].shape if len(prompt_embeddings[1]) >= 1 else None)
            print(len(video_embeddings), video_embeddings[1].shape if len(video_embeddings[1]) >= 1 else None)
        print(current_video_embedding.shape)
        print(current_prompt_embedding.shape)

        # print(filter_current_prompt)
        # print(filter_past_prompt_lst)
        # print()
        print('-------------------------------')
    print(dataset[1])