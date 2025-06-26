"""
加载pipline, 自回归的生成
"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../../'))
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from PIL import Image
from typing import Union, Optional, List, Tuple
import datetime

sys.path.append(os.path.join(os.getcwd(), '../../'))
from diffusers import CogVideoXDPMScheduler, CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
from Mutiple_prompt_mutiple_scene.pipline.CogVideoX_Muti_Prompt_pipline import CogVideoX_MultiPrompt_Pipeline
from Mutiple_prompt_mutiple_scene.utils import concat_videos, free_memory


def main(args):
    # 1. 加载pipline
    device = args.device if args.device is not None and ('cpu' == args.device or 'cuda' in args.device) else ('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = CogVideoX_MultiPrompt_Pipeline.from_pretrained_local(args.model_dir, device, args.dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    multi_prompts = args.multi_prompts

    # 用户输入过去视频
    multi_videos = args.multi_videos
    past_videos_lst = [] if multi_videos is None or len(multi_videos) == 0 else multi_videos[:]
    past_videos_lst = [video_path for video_path in past_videos_lst if os.path.exists(video_path)]
    past_prompts_lst = multi_prompts[:len(past_videos_lst)]

    # 输出文件夹
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, formatted_time)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 自回归生成
    for i in range(len(past_videos_lst), len(multi_prompts)):
        current_prompt = multi_prompts[i]
        if len(past_videos_lst) == 0:
            past_videos, past_prompts = None, None
        else:
            past_videos, past_prompts = past_videos_lst, past_prompts_lst
        save_video_path = os.path.join(output_dir, f'scene_{i}.mp4')
        pipe.generate_video(
            save_video_path=save_video_path,
            fps=args.fps,
            prompt=current_prompt,
            past_prompts=past_prompts,
            past_images=past_videos,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
            eta=args.eta,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            max_sequence_length=args.max_sequence_length
        )
        past_videos_lst.append(save_video_path)
        past_prompts_lst.append(current_prompt)
        free_memory()
    # 将视频拼接为长视频
    concat_videos(past_videos_lst, os.path.join(output_dir, "result.mp4"), fps=args.fps)
    print('Done !!!', '\n', f'video saved at: {os.path.join(output_dir, "result.mp4")}')


def get_parser():
    parser = argparse.ArgumentParser()

    # 加载模型
    parser.add_argument('--model_dir', type=str, default='../../excluded_dir/local_model/model_dir')

    # Pipline参数
    parser.add_argument('--multi_prompts', nargs='+', type=str, default=['Actor 1 is running on the playground.', 'Actor 2 is skiing on the snow.', 'Actor 3 is learning to run.'])
    parser.add_argument('--multi_videos', nargs='+', type=str, default=None)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--num_frames', type=int, default=49)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--guidance_scale', type=float, default=6.0)
    parser.add_argument('--image_guidance_scale', type=float, default=6.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--use_dynamic_cfg', type=bool, default=True)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--max_sequence_length', type=int, default=226)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=str, default='fp16', help='[fp16, bf16, fp32]')
    parser.add_argument('--seed', type=int, default=42)
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='../../excluded_dir/output_dir/inference/generate_video')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

"""
python inference.py \
--model_dir ../../excluded_dir/local_model/model_dir \
--multi_prompts 'Actor 1 is running on the playground.' 'Actor 2 is skiing on the snow.' 'Actor 2 is running on the playground.' \
--height 480 --width 720 --num_frames 49 --fps 15 \
--guidance_scale 6 --image_guidance_scale 6 \
--num_inference_steps 50 --device cuda --dtype fp16 --seed 42 \
--use_dynamic_cfg True --eta 0.0 \
--output_dir ../../excluded_dir/output_dir/inference/generate_video

"""

