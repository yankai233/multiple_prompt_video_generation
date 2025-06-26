# Multi-Scene video generation

## 1. 环境配置

### 1.1 下载第三方库

```python
pip install -r requirement.txt
```

### 1.2 下载预训练的模型：

|                   CLIP4CLIP(video_encoder)                   | https://huggingface.co/Searchium-ai/clip4clip-webvid150k     |
| :----------------------------------------------------------: | ------------------------------------------------------------ |
| **CogVideoX(tokenizer,text_encoder,transformer,scheduler,vae)** | [THUDM/CogVideoX-2b · Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b) |
|              **feature extractor transformer**               | [laion/CLIP-ViT-H-14-laion2B-s32B-b79K · Hugging Face](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |

将模型放到**Multi scene/excluded_dir/local_model/model_dir**，格式为：

```
model_dir
	|------Clip4Clip
	|
	+------CLIP-ViT-H-14-laion2B-s32B-b79K
	|
	+------feature_extractor
	|
	+------scheduler
	|
	+------text_encoder
	|
	+------tokenizer
	|
	+------transformer
	|
	+------transformer_t2v
	|
	+------vae
```

**请将CogVideoX中的transformer重命名为transformer_t2v**



## 2. 训练

### 2.1 feature extractor训练

#### 2.1.1 参数配置

详见**Multi scene/configs/Feature_Extraction_Module_T5.yaml**

#### 2.1.2 训练

执行以下命令训练：

```bash
python train_feature_extraction.py --log_dir ../../excluded_dir/output_dir/logs/train_feature_extraction --output_dir ../../excluded_dir/output_dir/logs/train_feature_extraction/ckpt --resume_from_checkpoint '' --configs ../../configs/Feature_Extraction_Module_T5.yaml --device cpu --mixed_precision fp16
```



### 2.2 CogVideoX训练

#### 2.2.1 参数配置

详见**Multi scene/Mutiple_prompt_mutiple_scene/Script/finetune/schemas/args.py**

#### 2.2.2 训练

执行以下命令**训练**：

```bash
bash train_CogVideoX_ti2v.sh
```



## 3. inference

