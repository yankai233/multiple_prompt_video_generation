import os
import sys

sys.path.append('../../')
sys.path.append('../../../')

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
from typing import Optional, Union, Tuple, List
from tqdm import tqdm
from pathlib import Path
import glob
import json

from vertexai.preview.generative_models import GenerativeModel, Part, Content, Image, ChatSession
from google.cloud import aiplatform
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import vertexai
import base64
import logging
import colorlog
import re
import time
from Mutiple_prompt_mutiple_scene.utils import path_join, save_json, load_json, path_apply_func
from typing import List, Optional, Union

import logging
import colorlog

# 配置日志
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
logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

MODEL_DICT = {'gemini-2.0': 'gemini-2.0-flash-001', 'gemini-2.0-lite': 'gemini-2.0-flash-lite-001',
              'gemini-embedding': 'gemini-embedding-001', 'text_embedding': 'text-embedding-005',
              'text-multilingual-embedding': 'text-multilingual-embedding-002',
              'multimodalembedding': 'multimodalembedding@001'}

SYSTEM_MESSAGE = """
You are a text processing assistant responsible for performing the following tasks:

1. **Text Cleaning**:

Clean the input [CLIP 1], [CLIP 2], and [current_prompt] by removing unnecessary words, but do not add or modify any words.
For example:
Input [CLIP 1]: Zhang San is in the dormitory → Cleaned as `Zhang San is in the dormitory.`
Input [current_prompt]: This video shows Li Si in the dorm → Cleaned as `Li Si is in the dorm.`
2. **Semantic Association Filtering**:

Based on the cleaned [current_prompt], filter out the tokens from the cleaned [CLIP 1], [CLIP 2]...,[CLIP N] that are semantically related to [current_prompt] and If a token appears in multiple [CLIP] inputs, only retain the token from the last [CLIP] where it appears..
For example:
当没有[CLIP]无需做语义筛选,只返回{'current_prompt': ...},不要加{'CLIP 1': ''}
[CLIP 1]The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 is holding the bag, and the bag is filled with objects. Actor1 opens the bag and says, "Hey everybody, Nick here, and today I got a review for you of this little guy. Um, this is the silent pocket uh 20-liter pack. Um first off though, I want to thank very much Silent...
[CLIP 2]The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 says "of course, here it is against your...
[current_prompt]The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 continues to say, \u201c...from afar, you know, it's freaking huge, so. But I mean, this is what it looks like
Cleaned [CLIP 1]: ``
Cleaned [CLIP 2]: indoors, view from above, beige carpet, black bag, Actor1
[current_prompt]: The scene is indoors with a view from above. The background is a beige carpet. A black bag is in the center of the scene. Actor1 continues to say, \u201c...from afar, you know, it's freaking huge, so. But I mean, this is what it looks like

3. **Note**:
Answer in English only, regardless of the language the user asks the question.

4. **Output Format**:

Return a dictionary in the following format:
{
  "CLIP 1": "token related to [current_prompt]",
  "CLIP 2": "token related to [current_prompt]",
  ...
  "CLIP N": "token related to [current_prompt]",
  "current_prompt": "cleaned [current_prompt]"
}

"""


def init_valid(vertex_client_json_path=r'E:\Data\datasets\Video_Datasets\Koala-36M\Code\APIServer\vertex_client.json',
               port=9002,
               project=None,
               token_json=None):
    """验证用户"""

    scopes = ['https://www.googleapis.com/auth/cloud-platform']
    if token_json is not None and os.path.exists(token_json):
        # TODO: 根据token验证
        creds = Credentials.from_authorized_user_file(token_json, scopes=scopes)
    else:
        # 加载OAuth 2.0 凭证
        flow = InstalledAppFlow.from_client_secrets_file(
            vertex_client_json_path, scopes=scopes
        )
        creds = flow.run_local_server(port=port)

    aiplatform.init(credentials=creds, project=project)


def extract_json_content(json_str):
    """抽取json内容"""

    # 定义正则表达式模式
    background_pattern = r'"background": "([^"]+)"'
    name_pattern = r'"name": "([^"]+)"'
    description_pattern = r'"description": "([^"]+)"'
    descriptions_pattern = r'"descriptions": \[([^\]]+)\]'

    # 提取数据
    background = re.search(background_pattern, json_str).group(1)
    names = re.findall(name_pattern, json_str)
    descriptions = re.findall(description_pattern, json_str)
    descriptions_list = re.search(descriptions_pattern, json_str).group(1)

    # 动态构建 characters 列表
    characters = []
    for i in range(len(names)):
        characters.append({"name": names[i], "description": descriptions[i]})

    # 动态构建 descriptions 列表
    descriptions = [desc.strip().strip('"') for desc in descriptions_list.split(",")]

    # 构建字典
    parsed_data = {
        "background": background,
        "characters": characters,
        "descriptions": descriptions
    }

    return parsed_data


def video2part(video_path):
    """视频转为模型输入"""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    video_part = Part.from_data(data=video_base64, mime_type='video/mp4')
    return video_part


def load_local_data(content_lst):
    """加载数据"""
    res_content_lst = []
    if isinstance(content_lst, str):
        content_lst = [content_lst]
    for content in content_lst:
        try:
            if os.path.exists(content):
                # 图像
                if os.path.basename(content).split('.')[-1] in ['jpg', 'png', 'jpeg']:
                    image = Image.load_from_file(content)
                    res_content_lst.append(image)
                    logger.debug(f'图像:{content}')
                # 视频
                elif os.path.basename(content).split('.')[-1] in ['mp4']:
                    video = video2part(content)
                    res_content_lst.append(video)
                    logger.debug(f'视频:{content}')
                else:
                    raise ValueError(f'输入路径文件错误: {content}')
            # 文本
            else:
                res_content_lst.append(content)
                logger.debug(f'文本: {content}')
        except Exception as e:
            logger.error(f'load local file error: {e}')
    return res_content_lst


def chat_describe_videos(
        model_input_lst: List[Part],
        video_path_lst: List[str],
        chat: ChatSession,
        start_prompt=None,
        show_progress=True,
        prompts_dict=None,
        save_path=None,
        overwrite=False,
        response_func=None,
        init_valid_func=None,
        max_suq_len=10
):
    """
    :param model_input_lst:     video Part list
    :param video_path_lst:      video path list
    :param chat:                chat session
    :param prompts_dict         预测结果
    :param save_path            保存路径
    :param overwrite            覆盖prompts_dict
    :param response_func        response_func(response.text)
    :param init_valid           若API 需要reflash,重新验证
    :return:
        answer_lst
    """
    # 处理输入
    response_func = (lambda x: x) if response_func is None else response_func
    model_input_content_lst = model_input_lst[:max_suq_len] if len(model_input_lst) > max_suq_len else model_input_lst[
                                                                                                       :]
    video_path_lst = video_path_lst[:max_suq_len] if len(video_path_lst) > max_suq_len else video_path_lst[:]
    model_input_content_lst = tqdm(model_input_content_lst) if show_progress else model_input_content_lst
    start_prompt = '请描述下面视频' if start_prompt is None else start_prompt
    prompts_dict = prompts_dict if prompts_dict is not None and not overwrite else {}

    error_count = 0
    for step, model_input in enumerate(model_input_content_lst):
        video_path = video_path_lst[step]
        # video1/split/clip1.mp4
        video_key = path_apply_func(video_path, func=lambda x: x[-3:])
        if video_key in prompts_dict:
            logger.info(f'skip: {video_key}')
            continue
        for i in range(20):
            try:
                response = chat.send_message([start_prompt, model_input])
                prompts_dict[video_key] = response_func(response.text)
                # 保存
                if save_path is not None:
                    os.makedirs(path_apply_func(save_path, lambda x: x[:-1]), exist_ok=True)
                    save_json(save_path, prompts_dict)
                break
            # 报错
            except Exception as e:
                logger.error(f'Error: chat_describe_videos \t{video_path}: {e}')
                # 重新验证
                if init_valid_func is not None and i == 0:
                    try:
                        init_valid_func()
                    except:
                        pass

                time.sleep(0.5)
    return prompts_dict


def get_gemini_prompt_from_past_and_current(
        current_prompt: str,
        past_prompt_lst: Optional[List[str]] = None
) -> str:
    """
    根据current_prompt和past_prompt_lst得到gemini的输入提示
    :returns
        "
            past_prompt_tokens:[CLIP 1] 描述1\n[CLIP 2] 描述2\n
            current_prompt:...
        "
    """
    if past_prompt_lst is None or len(past_prompt_lst) == 0:
        return f'current_prompt:{current_prompt}'
    past_prompt_tokens = ''
    for step, past_prompt in enumerate(tqdm(past_prompt_lst, desc='chat feature filter')):
        past_prompt_tokens = past_prompt_tokens + f'[CLIP {step + 1}]' + past_prompt + '\n'
    gemini_prompt = f'past_prompt_tokens:{past_prompt_tokens}\ncurrent_prompt:{current_prompt}'
    return gemini_prompt


def response_func(json_str):
    """
    根据json str解析出dict
    """
    # TODO
    pattern = r'```json\s*({[\s\S]*?})\s*```'
    match = re.search(pattern, json_str)
    if match:
        json_content = match.group(1)
        extract_dict = json.loads(json_content)
        for k, v in extract_dict.items():
            if v is None:
                extract_dict[k] = ''
        return extract_dict
    logger.error('匹配失败')


def get_token_index_lst(filter_prompt_lst, past_prompt_lst, current_prompt):
    """
    :param filter_prompt_lst:         [filter_prompt1, filter_prompt2, ...]
    :param past_prompt_lst:
    :return:

    """
    past_prompt_lst = [] if past_prompt_lst is None else past_prompt_lst
    all_prompt = past_prompt_lst + [current_prompt]
    res = []
    for filter_prompt, prompt in zip(filter_prompt_lst, all_prompt):
        indexs = get_token_index(prompt, filter_prompt)
        res.append(indexs)
    return res

def get_token_index(str1, str2):
    """查询str2的token在str1的token位置"""
    str1_lst = str1.split(' ')
    str2_lst = str2.split(' ')
    res = []
    for token2 in str2_lst:
        try:
            idx = str1_lst.index(token2)
            res.append(idx)
        except Exception as e:
            res.append(-1)
            logger.warning(f'在str1: `{str1}` 中未查询到str2的token:{token2}')
    return res


def chat_feature_filter(
        # chat输入
        current_prompt: str,
        model: GenerativeModel,
        start_prompt: Optional[str] = None,
        past_prompt_lst: Optional[List[str]] = None,
        # chat输出
        response_func=None,
        # 结果保存
        response_dict: Optional[dict] = None,
        save_path: Optional[str] = None,
        overwrite: bool = False,
        # 错误处理
        init_valid_func=None,
        max_error_time: int = 2,
        *args,
        **kwargs
):
    """
    根据current_prompt筛选出past_prompt_lst中的token
    :returns
        {'gemini_prompt': {'CLIP 1': filter_tokens, 'CLIP 2': filter_tokens, 'current_prompt': output_current_prompt}}
    """
    # 1. 处理输入
    past_prompt_lst = past_prompt_lst if past_prompt_lst is not None else []
    if past_prompt_lst is None or len(past_prompt_lst) == 0:
        past_prompt_lst = None
    start_prompt = start_prompt if start_prompt is not None and len(
        start_prompt) != 0 else '删除past_prompt_tokens中与current_prompt的单词无关的单词,并输出未被删除的单词'
    response_func = (lambda x: x) if response_func is None else response_func
    response_dict = response_dict if response_dict is not None else {}
    # 2. 过去文本
    gemini_prompt = get_gemini_prompt_from_past_and_current(current_prompt=current_prompt,
                                                            past_prompt_lst=past_prompt_lst)
    if gemini_prompt in response_dict.keys() and not overwrite:
        return response_dict
    # 3. 模型预测

    for error_time in range(max_error_time):
        try:
            response = model.generate_content([start_prompt, gemini_prompt])
            # 解析: {'CLIP 1': ...,'CLIP 2': ...,'current_prompt': output_current_prompt}
            filter_tokens = response_func(response.text)
            response_dict[gemini_prompt] = filter_tokens
            if save_path is not None:
                save_json(save_path, response_dict)
            return response_dict
        except Exception as e:
            if init_valid_func is not None and error_time == 0:
                init_valid_func()
            logger.error(f'Error: chat_feature_filter error: {e}')
            time.sleep(0.5)


class Gemini_API:
    def __init__(self, vertex_client_json_path, port, model_name, system_message=None, project=None):
        self.vertex_client_json_path = vertex_client_json_path
        self.port = port
        self.model_name = model_name
        if model_name not in MODEL_DICT:
            logger.warning(f'你的model_name为:{model_name}, 只能从{list(MODEL_DICT.keys())}中选择,自动选择gemini-2.0-lite模型')
            self.model_name = 'gemini-2.0-lite'
        self.system_message = SYSTEM_MESSAGE if system_message is None or len(system_message) == 0 else system_message
        self.project = project
        # 身份验证
        self.flash_token()

    def flash_token(self):
        init_valid(self.vertex_client_json_path, self.port, self.project)
        self.model = GenerativeModel(MODEL_DICT[self.model_name], system_instruction=self.system_message)

    def change_system_message(self, new_system_message):
        self.system_message = new_system_message
        self.flash_token()

    def chat_feature_filter(self,
                            # chat输入
                            current_prompt: str,
                            start_prompt: Optional[str] = None,
                            past_prompt_lst: Optional[List[str]] = None,
                            # chat输出
                            response_func=None,
                            # 结果保存
                            response_dict: Optional[dict] = None,
                            save_path: Optional[str] = None,
                            overwrite: bool = False,
                            # 错误处理
                            max_error_time: int = 2,
                            *args,
                            **kwargs
    ):
        """
        :return:
            response_dict: {
                '原文本': {'CLIP 1':, 'CLIP 2', 'current_prompt':}
            }
            idxs_lst: [token_index(原文本,筛选文本)]
        """
        result_dict = chat_feature_filter(
            current_prompt=current_prompt,
            model=self.model, start_prompt=start_prompt, past_prompt_lst=past_prompt_lst,
            response_func=response_func, response_dict=response_dict, save_path=save_path, overwrite=overwrite,
            init_valid_func=lambda: init_valid(self.vertex_client_json_path, self.port, self.project),
            max_error_time=max_error_time, *args, **kwargs
        )

        return result_dict


# 用户输入
json_path = r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\excluded_dir\dataset\prompts\0-ggn3z52oU_76.json'

# 模型类型
model_name = MODEL_DICT['gemini-2.0-lite']
# 指令

system_message = SYSTEM_MESSAGE

if __name__ == '__main__':
    d: dict = load_json(json_path)
    past_prompt_lst = list(d.values())[:-1]
    current_prompt = list(d.values())[-1]

    # 加载模型
    api = Gemini_API('./vertex_client.json', 9002, model_name='gemini-2.0-lite',system_message=system_message, project='video-describe')
    response_dict = api.chat_feature_filter(current_prompt=current_prompt, past_prompt_lst=past_prompt_lst, response_func=response_func, response_dict=None, save_path='./1.json', overwrite=False)
    response_lst = list(response_dict[get_gemini_prompt_from_past_and_current(current_prompt, past_prompt_lst)].values())
    index_lst = get_token_index_lst(response_lst, past_prompt_lst, current_prompt)
    print(index_lst)

