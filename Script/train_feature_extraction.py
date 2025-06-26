"""
训练特征抽取模块
"""
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))

from tqdm.contrib.logging import logging_redirect_tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
from transformers import (AutoTokenizer, T5EncoderModel, CLIPVisionModel, CLIPProcessor, get_scheduler, CLIPTokenizer, CLIPTextModel, 
            CLIPTextModelWithProjection, CLIPVisionModelWithProjection)
from omegaconf import OmegaConf

from Mutiple_prompt_mutiple_scene.utils import *
from Mutiple_prompt_mutiple_scene.Model.transformer_blocks import *
from Mutiple_prompt_mutiple_scene.Mydataset.MyDataset import Feature_Extraction_Dataset, collate_fn
from Mutiple_prompt_mutiple_scene.Script.APIServer.Gemini_API import Gemini_API

logger = init_logging('../../excluded_dir/output_dir/logs/1/train_fe_log.txt', __name__)
mixed_precision_dict = ['no', 'fp16', 'bf16']


def check_inputs(args, configs):
    """检查输入,并创建文件夹"""
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset_root = configs.data.train.train_root
    val_dataset_root = configs.data.valid.valid_root

    assert os.path.exists(train_dataset_root), f'数据集路径不存在,configs.data.train.train_root:{train_dataset_root}'
    assert val_dataset_root == '' or val_dataset_root is None or os.path.exists(val_dataset_root), f'验证集路径不存在,configs.data.valid.valid_root: {val_dataset_root}'


def resume_config_ckpt(args):
    """
    加载ckpt
    :returns
        resume, configs
    """
    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != '' and os.path.exists(args.resume_from_checkpoint):
        resume = args.resume_from_checkpoint
        parent_dir = os.path.dirname(resume)
        configs = os.path.join(parent_dir, 'config.yaml')
    else:
        resume = ''
        configs = args.configs

    return resume, OmegaConf.load(configs)


def cosine_similarity_loss(token1, token2):
    """
    计算余弦相似度损失。
    :param token1: 形状为 [b, 77, 512]
    :param token2: 形状为 [b, 77, 512]

    """
    # 对 token 进行归一化
    token1 = F.normalize(token1, dim=-1)  # 形状为 [b, 226, 4096]
    token2 = F.normalize(token2, dim=-1)  # 形状为 [b, 226, 4096]

    # 计算余弦相似度
    cosine_sim = torch.einsum('bik,bjk->bij', token1, token2)  # 形状为 [b, 226, 226]
    loss = 1 - cosine_sim.mean()
    return loss


def save_model_config_history(feature_extraction_model, configs, history, output_dir, last_name='last', info=None):
    if info is not None:
        print(info)
    print('保存模型和数据中...')
    torch.save(feature_extraction_model.state_dict(), os.path.join(output_dir, 'feature_extraction_model_{}.pth'.format(last_name)))
    OmegaConf.save(configs, os.path.join(output_dir, 'config.yaml'))
    save_json(os.path.join(output_dir, f'history_{last_name}.json'), history)
    print('保存完成')


@torch.no_grad()
def validation(feature_extraction_model, video_encoder, text_encoder, tokenizer, processor,
               valid_dataloder, args, configs, device=None):
    """验证"""
    print('验证中...')
    device = device if device == 'cpu' or 'cuda' in device else ('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extraction_model.eval()
    video_encoder.eval()
    text_encoder.eval()
    loss_avgmeter_valid = AverageMeter()

    with logging_redirect_tqdm(loggers=[logger]):
        process_bar = tqdm(valid_dataloder, desc='Feature Extraction Module Valid')
        for step, batch in enumerate(process_bar):
            video_embeddings, text_embeddings, current_text_embeddings, current_video_embeddings = batch[
                'video_embeddings'], batch['prompt_embeddings'], batch['current_prompt_embedding'], batch[
                'current_video_embedding']
            image_embedding_batch = listtensor_to_device(video_embeddings, device)
            text_embedding_batch = listtensor_to_device(text_embeddings, device)
            current_text_embedding_batch = current_video_embeddings.to(device)
            current_video_embedding_batch = current_video_embeddings.to(device)
            # pred_feature: [b,l,512]
            pred_feature = feature_extraction_model(image_embedding_batch, text_embedding_batch,
                                                    current_text_embedding_batch)
            # real_feature: [b,l,512]
            real_feature = feature_extraction_model.no_transformer(current_video_embedding_batch,
                                                                       current_text_embedding_batch)
            # 计算损失
            loss = cosine_similarity_loss(real_feature.float(), pred_feature.float())

            # 更新进度条 + 评估
            loss_avgmeter_valid.update(loss.item(), 1)
            process_bar.set_postfix(loss=loss.item())
            process_bar.update(1)

    print('验证完成')
    return loss_avgmeter_valid.avg


def main(args):

    device = args.device if args.device == 'cpu' or 'cuda' in args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    resume, configs = resume_config_ckpt(args)
    model_params, data_params, train_params = configs.model, configs.data, configs.train_params
    check_inputs(args, configs)

    # 1. 加载模型
    load_params = model_params.load_params
    model_id = load_params.pretrained_model_path
    subfolder_video_encoder, subfolder_text_encoder, subfolder_tokenizer = load_params.subfolder_video_encoder, load_params.subfolder_text_encoder, load_params.subfolder_tokenizer
    local_files_only = True if os.path.exists(model_id) else False
    print('加载模型中...')
    feature_extraction_model = load_feature_extractor_from_ckpt(model_params.Feature_Extraction_Module, resume)

    if subfolder_text_encoder == 'Clip4Clip':
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder=subfolder_tokenizer,
                                                  local_files_only=local_files_only)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder=subfolder_text_encoder,
                                                     local_files_only=local_files_only).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder_tokenizer, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder=subfolder_text_encoder, local_files_only=local_files_only).to(device)

    video_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder=subfolder_video_encoder, local_files_only=local_files_only).to(device)
   
    # 设置梯度
    feature_extraction_model.requires_grad_(True)
    text_encoder.requires_grad_(False)
    video_encoder.requires_grad_(False)

    # to(device)
    feature_extraction_model = feature_extraction_model.to('cpu')
    text_encoder = text_encoder.to(device)
    video_encoder = video_encoder.to(device)
    print('模型加载完成')

    # Gemini API
    gemini_params = model_params.Feature_filter_Gemini_API
    gemini_api = None
    if gemini_params.use_feature_extractor:
        gemini_api = Gemini_API(gemini_params.init_valid.vertex_client_json_path, gemini_params.init_valid.port, gemini_params.model_name, system_message=gemini_params.system_prompt, project=gemini_params.project)

    # 2. 数据集
    cache_dir_name = os.path.basename(args.configs).split('.')[0]
    train_dataset = Feature_Extraction_Dataset(data_params.train.train_root, data_params.train.height, data_params.train.width, logger=logger,
                                               video_encoder=video_encoder,
                                               text_encoder=text_encoder, tokenizer=tokenizer,
                                               gemini_api=gemini_api, use_feature_extractor=gemini_params.use_feature_extractor,
                                               device=device, cache_dir_name=cache_dir_name, api_start_prompt=gemini_params.start_prompt,
                                               num_frames=data_params.train.train_num_frames)
    print('正在预处理训练集数据中(encode image and prompt)...')
    train_dataloader = data.DataLoader(train_dataset, batch_size=1,
                                       collate_fn=collate_fn)
    for i, batch in enumerate(tqdm(train_dataloader, desc='data prepare')):
        try:
            pass
        except:
            logger.error(f'第{i}个数据加载错误: {train_dataset.videos_regress[i]}')
    train_dataloader = data.DataLoader(train_dataset, batch_size=data_params.train.train_batch_size,
                                       num_workers=data_params.train.num_workers,
                                       shuffle=data_params.train.shuffle,
                                       collate_fn=collate_fn
                                       )
    is_valid = os.path.exists(data_params.valid.valid_root)
    print('加载训练集完成')
    if is_valid:
        valid_dataset = Feature_Extraction_Dataset(data_params.valid.valid_root, data_params.valid.height, data_params.valid.width, logger=logger,
                                               video_encoder=video_encoder, text_encoder=text_encoder, tokenizer=tokenizer,
                                               device=device, cache_dir_name=cache_dir_name)

        print('正在预处理测试集数据中(encode image and prompt)...')
        valid_dataloder = data.DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn)
        for batch in valid_dataloder:
            ...
        valid_dataloder = data.DataLoader(valid_dataset, batch_size=data_params.valid.valid_batch_size, num_workers=data_params.valid.num_workers, shuffle=data_params.valid.shuffle, collate_fn=collate_fn)
        print('加载验证集完成')
    # 卸载image encoder 和 text encoder
    video_encoder = video_encoder.to('cpu')
    text_encoder = text_encoder.to('cpu')
    free_memory()

    # 3. 优化器
    print('加载优化器中...')
    optimizer = optim.AdamW(feature_extraction_model.parameters(),
                            lr=train_params.optim.lr, weight_decay=train_params.optim.weight_decay,
                            betas=(train_params.optim.adam_beta1, train_params.optim.adam_beta2),
                            eps=train_params.optim.adam_epsilon)
    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer,
        num_warmup_steps=train_params.optim.lr_warmup_steps,
        num_training_steps=train_params.max_epochs * len(train_dataloader)
    )
    print('加载优化器完成')

    # 4. train
    print('<================train!!!================>')
    history = {'loss': []}
    best_history = float('inf')
    global_step = 0
    if gemini_params.use_feature_extractor:
        padding_embedding = torch.zeros((model_params.Feature_Extraction_Module.text_embed_dim), requires_grad=False)
    for epoch in range(train_params.max_epochs):
        feature_extraction_model = feature_extraction_model.train(True).to(device)
        loss_avgmeter = AverageMeter()
        process_bar = tqdm(train_dataloader, desc='Feature Extraction Module Training')
        for step, batch in enumerate(process_bar):
            if gemini_params.use_feature_extractor:
                from Mutiple_prompt_mutiple_scene.Model.feature_filter import feature_filter
                video_embeddings, text_embeddings, current_text_embeddings, current_video_embeddings, meta_info = batch['video_embeddings'], batch['prompt_embeddings'], batch['current_prompt_embedding'], batch['current_video_embedding'], batch['meta_info']
                filter_past_token_index = meta_info['filter_past_token_index']
                filter_current_token_index = meta_info['filter_current_token_index']
                text_embeddings, current_text_embeddings = feature_filter(text_embeddings, current_text_embeddings, filter_past_token_index, filter_current_token_index, padding_embedding=padding_embedding, max_suq_len=model_params.Feature_Extraction_Module.text_squ_len)
            else:
                video_embeddings, text_embeddings, current_text_embeddings, current_video_embeddings, meta_info = batch['video_embeddings'], batch['prompt_embeddings'], batch['current_prompt_embedding'], batch['current_video_embedding'], batch['meta_info']
            video_embedding_batch = listtensor_to_device(video_embeddings, device)
            text_embedding_batch = listtensor_to_device(text_embeddings, device)
            current_text_embedding_batch = current_text_embeddings.to(device)
            current_video_embedding_batch = current_video_embeddings.to(device)

            # pred_feature: [b,l,512]
            feature_extraction_model.zero_grad()
            pred_feature = feature_extraction_model(video_embedding_batch, text_embedding_batch, current_text_embedding_batch)
            # real_feature: [b,l,512]
            with torch.no_grad():
                real_feature = feature_extraction_model.no_transformer(current_video_embedding_batch, current_text_embedding_batch)
            # 计算损失
            loss = cosine_similarity_loss(real_feature.float(), pred_feature.float())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # 更新进度条 + 评估
            loss_avgmeter.update(loss.item(), 1)
            process_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            process_bar.update(1)
            global_step += 1
            # 保存模型 + config + history
            if global_step % train_params.save_times == 0 and global_step > 0:
                save_model_config_history(feature_extraction_model, configs, history, args.output_dir, info=f'global_step={global_step}, 数据保存中...')

        history['loss'].append(loss_avgmeter.avg)
        print(f'{epoch}/{train_params.max_epochs - 1}, loss: {loss_avgmeter.avg}')
        # TODO: 验证集验证
        if is_valid:
            del loss
            free_memory()
            valid_res = validation(feature_extraction_model, video_encoder, text_encoder, tokenizer, valid_dataloder, args, configs, device)
            if valid_res <= best_history:
                best_history = valid_res
                # 保存
                save_model_config_history(feature_extraction_model, configs, history, args.output_dir, 'best', info='保存best model中...')
            free_memory()
        # 保存
        if epoch % train_params.save_epoch_times == 0 or epoch == train_params.max_epochs - 1:
            save_model_config_history(feature_extraction_model, configs, history, args.output_dir, 'last', info=f'epoch={epoch}, 保存数据中...')


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='../excluded_dir/output_dir/logs/1')
    parser.add_argument('--output_dir', type=str, default='../excluded_dir/output_dir/logs/1/ckpt')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--configs', type=str, default='../../configs/Feature_Extraction_Module_clip.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # args.configs = r'E:\PythonLearn\work\SSH_Connect\Autodl\under2postgraudate\Video-Generation-field\Ours\Multiple scene\Mutiple_prompt_mutiple_scene\configs\Feature_Extraction_Module_clip.yaml'
    main(args)

"""
python train_feature_extraction.py --log_dir ../../excluded_dir/output_dir/logs/train_feature_extraction --output_dir ../../excluded_dir/output_dir/logs/train_feature_extraction/ckpt --resume_from_checkpoint '' --configs ../../configs/Feature_Extraction_Module_T5.yaml --device cpu --mixed_precision fp16

"""
