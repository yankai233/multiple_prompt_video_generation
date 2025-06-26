import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))
sys.path.append(os.path.join(os.getcwd(), '../../'))

from omegaconf import OmegaConf
from typing import List, Tuple, Any, Optional, Union
import torch
import os
import glob
import json
import gc
import logging
import colorlog
import platform
from pathlib import Path
import cv2
from moviepy import VideoFileClip, concatenate_videoclips

from diffusers import CogVideoXTransformer3DModel

from Mutiple_prompt_mutiple_scene.Model.CogVideoX_ti2v import CogVideoXTransformer3D_TI2V
from Mutiple_prompt_mutiple_scene.Model.transformer_blocks import Feature_Extraction_Module


train_layer_name_lst = []

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def init_logging(file_path=None, name=None):
    # 配置日志
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            handlers=[
                # 把日志同时输出到文件和控制台
                logging.FileHandler(file_path, mode='w', encoding='utf-8'),
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
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
    logger = logging.getLogger(name)
    logger.addHandler(console_handler)
    return logger


def path_join(*args):
    path = args[0]
    for i in range(1, len(args)):
        path = os.path.join(path, args[i])
    return path.replace('\\', '/')


def path_apply_func(path, func):
    """根据path获得key"""
    return '/'.join(func([s for s in path_join(path).split('/') if len(s) != 0]))


def path_replace(path, replace_path, start, end='inf'):
    """
        example: path='E:\PythonLearn\work\SSH_Connect\Autodl\graduation_thesis\'
                 replace_path='lr'
                 start=-2, end='inf'
        -> 'E:\PythonLearn\work\SSH_Connect\lr'
    """
    split_lst = [s for s in path_join(path).split('/') if len(s) != 0]
    if end == 'inf':
        split_lst[start:] = [replace_path]
    else:
        split_lst[start: end] = [replace_path]
    path = '/'.join(split_lst)
    return path


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_json(file_path, d):
    # 保存数据
    with open(file_path, 'w') as f:
        json.dump(d, f, indent=4)


def load_json(file_path):
    # 加载数据
    with open(file_path, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


def get_path_str(path):
    return path if isinstance(path, str) else str(path)


def load_transformer_multi_ti2v(model_path, subfolder='transformer', local_files_only=True):
    """
    加载transformer ti2v模型
        若存在.safetensor文件，则自接加载
        若不存在，则导入transformer_t2v的部分参数作为预训练初始化
    :returns
        CogVideoXTransformer3D_TI2V
    """
    if not local_files_only:
        print('从huggingface下载transformer')
        return CogVideoXTransformer3D_TI2V.from_pretrained(model_path, subfolder=subfolder)
    if glob.glob(os.path.join(model_path, subfolder, '*.safetensors')):
        print('本地加载预训练的transformer')
        return CogVideoXTransformer3D_TI2V.from_pretrained(model_path, subfolder=subfolder, local_files_only=local_files_only, use_safetensors=True)
    print('本地加载transformer, 根据transformer_t2v初始化')
    config_path = os.path.join(model_path, subfolder, 'config.json')
    transformer_config = load_json(config_path)
    transformer_ti2v = CogVideoXTransformer3D_TI2V(**transformer_config)
    transformer_t2v = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder='transformer_t2v', local_files_only=local_files_only)
    # 加载transformer_t2v参数
    transformer_ti2v.load_state_dict(transformer_t2v.state_dict(), strict=False)
    del transformer_t2v
    return transformer_ti2v


def load_feature_extractor_from_ckpt(Feature_Extraction_Module_params: Union[str, OmegaConf], resume: str):
    """
    加载特征抽取组件
    Feature_Extraction_Module_params: OmegaCofig | path: str
    resume: str
    :returns
        Feature_Extraction_Module
    """
    Feature_Extraction_Module_params = OmegaConf.load(Feature_Extraction_Module_params).model.Feature_Extraction_Module if isinstance(Feature_Extraction_Module_params, str) else Feature_Extraction_Module_params
    fxm = Feature_Extraction_Module(**Feature_Extraction_Module_params)
    if resume is not None and resume != '' and os.path.exists(resume):
        print(f'从{resume}加载Feature_Extraction_Module')
        fxm.load_state_dict(torch.load(resume), strict=False)
    else:
        print('Feature_Extraction_Module随机初始化')
    return fxm


def get_long_path(path):
    path = get_path_str(path)
    if platform.system() == 'Windows':
        return fr'\\?\{path}'
    return path


def free_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def listtensor_to_device(lst_tensor: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    """[tensor('cpu')] -> [tensor('cuda')]"""
    for i in range(len(lst_tensor)):
        tensor = lst_tensor[i]
        if len(tensor) == 0:
            continue
        elif isinstance(tensor, torch.Tensor):
            lst_tensor[i] = tensor.to(device)
        elif isinstance(tensor, list):
            lst_tensor[i] = torch.tensor(tensor).to(device)
    return lst_tensor


def concat_videos(video_files: List[str], output_video_path: str, fps: Optional[int] = None) -> None:
    """将多个视频拼接为长视频"""
    try:
        # 读取视频文件
        clips = [VideoFileClip(video) for video in video_files]

        # 拼接视频
        final_clip = concatenate_videoclips(clips)
        # 保存拼接后的视频
        fps = clips[0].fps if fps is None else fps
        final_clip.write_videofile(output_video_path, fps=fps)
        # 关闭视频剪辑对象以释放资源
        for clip in clips:
            clip.close()
        final_clip.close()
    except Exception as e:
        print('ERROR: ', str(e))
