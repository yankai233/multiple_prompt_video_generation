# Multi-Scene long video generation

由于数据集稀少以及算力原因，本工作目前还属于遗弃阶段。我目前只训练了3天，发现场景是可以保持一致性的。

给定的4个文本描述分别为：

1. 'Actor 1 is running on the playground.'

2. 'Actor 2 is skiing on the snow.' 

3. 'Actor 2 is running on the playground.'

4.  'Actor 1 is skiing on the snow.'

   生成的视频效果：

   <video width="720" height="480" controls>
       <source src="./asset/result.mp4" type="video/mp4">
   </video>

   其中，可以看到场景基本保持一致，人物还没有保持一致，但这很有可能是训练不充分以及数据集的问题，需要注意的是这只在1000个视频、vGPU(48GB)上训练了3天达到的效果，也可以看到，我们的方法确实在尽可能的保证人物的一致性，第3个场景的人物部分是红色衣服，与第2个场景的红色衣服对应。第4个场景的人物穿了绿衣服与第一个场景的绿衣服对应。

   如果你能接受这一结果，后续将向您介绍完整的pipline，并且我们提供了完整可训练的代码。

   【注】如果您使用了我的Pipline并进行了部分修改，且想要发paper，希望能挂我个3、4、5作（谢谢，这对研0的我至关重要:joy:）。联系我：3475287084@qq.com，QQ：3475287084，备注您的来意，谢谢，谢谢，谢谢:kissing_heart::kissing_heart::kissing_heart:。

   ## 1. pipline介绍

![image-20250628103157341](./asset/pipline.png)

我们的任务是根据过去的文本提示${\{P_1,P_2,...,P_{i-1}\}}$以及视频片段${\{V_1,V_2,...,V_{i-1}\}}$以及当前的文本提示$P_i$自回归的生成$V_i$，即：$V_i=G(\{V_1,P_1\},\{V_2,P_2\},...,\{V_{i-1},P_{i-1}\},P_i)$。我们的pipline分为4个模块：文本筛选模块、文本视频注意力抽取模块、文本视频解耦模块、视频生成模块。

## 1.1 文本筛选模块

随着$V_i$的生成，当i很大时，模型容易产生遗忘重要信息，与FramePack和WorldMEM[4]的想法相同，与其让模型被动遗忘重要信息，不如让模型主动遗忘无用信息，这就是文本筛选模块的目标。

这是个NLP任务，目标是根据P~i~的toekn筛选出P~1~,...,P~i-1~ 中与P~i~  token语义最接近的token，如：P~1~=`猫在开车`，P~2~=`一只狗在骑自行车`，P~3~=`一只猫在骑自行车`。文本筛选模块会计算P~3~的token与P~1~和P~2~的token之间的相似度，然后通过阈值的方法，筛选出过去对P~3~有帮助的信息，如P~1~的’猫’和P~2~的自行车对应的token。

### 1.2 文本对应的视频语义（注意力）抽取模块

我们需要在更细粒度的token级别而非Prompt级别进行视频语义抽取，即根据过去有用的token {T~1~,T~2~,...,T~k~}及token对应的视频片段{V~1~,V~2~,...,V~k~}抽取P~i~的所有token对应的视觉语义信息。

方法：

\1.  {T1,T2,...,Tk}与{V1,V2,...,Vk}分别与做交叉注意力

$$
Attention(Q,K,V)=Sotfmax({\frac{QK^T}{\sqrt d}})·V
$$
其中，$Q=W_Q(\tau(T_i))$，$K=W_K(\phi(V_i))$，$V=W_V(\phi(V_i))$，$\tau$为文本编码器，$\phi$为视频编码器。

\2. 将$Attention_i$与$\tau(T_i)$在最后一个维度拼接,记作$\Phi_i=concat([Attention_i,\tau(T_i)])$。然后将所有token在token维度拼接$\Phi=\{\Phi_1,\Phi_2,...,\Phi_{i-1}\}$输入到transformer中抽取出的P~i~所有token与目标视频片段对应的交叉注意力$[Attention_{k+1},Attention_{k+2},...]$

\3. 损失函数

由于该模块的输出并不是完全的视频视觉信息，而是文本与视觉的交叉注意力，因此损失函数被设计为：

$$
Attention_{real}=Softmax({\frac {QK^T} {\sqrt d}})·V\\
Attention_{pred}=transformer(\Phi,P_i)\\
L=Loss(Attention_{real}-Attention_{pred})
$$
其中，在代码中，这里的Loss目前选择的是1-余旋相似度。这个模块与视频生成无关，因此被单独训练，并且在后续被冻结。

### 1.3 文本语义和视频语义解耦模块

由于1.2的结果为token与视频之间的交叉注意力，将其直接与文本语义拼接会导致文本与视频不解耦，因此这里我们可以设计一个解耦网络，我目前设计的解耦网络就是让文本与注意力图再做一次交叉注意力，让文本语义关注交叉注意力图抽取出视频语义。该模块与CogVideoX一起训练。当然这个模块被设计的初衷是要做解耦的，但目前还没有设计损失函数来约束其解耦效果

### 1.4 多模态注入

直接将文本embedding和video embedding拼接输入到CogVideoX中。

### 1.5 新加的Classifier-free guidance

在我们的pipline中，新加了Classifier-free guidance（参考至History Guidance）：

$$
\epsilon_θ (z_t,t,\{V_1,P_1^+\},…,\{V_{i-1},P_{i-1}^+\},P_i^+)=ϵ_θ (z_t,t,\{V_1,P_1^+ \},…,\{V_{i-1},P_{i-1}^+\},P_i^- )+w_1*(ϵ_θ (z_t,t,\{V_1,P_1^+\},…,\{V_{i-1},P_{i-1}^+ \},P_i^+ )-ϵ_θ (z_t,t,\{V_0,P_0^+ \},…,\{V_{i-1},P_{i-1}^+ \},P_i^- ))+w_2*(ϵ_θ (z_t,t,\{V_0,P_0^+ \},…,\{V_{i-1},P_{i-1}^+ \},P_i^+ )-ϵ_θ (z_t,t,P_i^+ ))
$$
其中，P^+^表示正向词，P^-^表示反向词，该前2项为普通的Classifier-free guidance，最后一项确保了当前视频的生成能更好的依赖于过去视频和文本描述。

## 2. 环境配置

```bash
git clone git@github.com:yankai233/multiple_prompt_video_generation.git
```

### 2.1 下载第三方库

```bash
pip install -r requirement.txt
cd Script
```

### 2.2 下载预训练的模型：

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

## 3. 训练

### 3.1 数据集配置

创建数据集部分请参考**论文**：[[2503.19881\] Mask$^2$DiT: Dual Mask-based Diffusion Transformer for Multi-Scene Long Video Generation](https://arxiv.org/abs/2503.19881)

```markdown
1. 从youtubu下载视频,这里我使用Koala-36M的链接下载视频
2. 将视频裁剪为10min内
3. 使用PySceneDetect将视频分段
4. 使用Gemini大模型描述视频内容（Gemini可以连续描述10轮视频，因此只能保证10个片段的描述一致）
5. 数据集格式：
data_root
	|
	+-------videos
	|		   |-------video1
	|		   |		|--------片段1
	|		   |		|--------片段2
	|		   |
	|		   |-------video2
	|		   
	|
	+-------prompts
			   |--------video1.json
			   |--------video2.json
			   
[注]:video1,json内容为:{片段1路径:描述,片段2路径:描述}
	
```

### 3.1 feature extractor训练

#### 3.1.1 参数配置

详见**Multi scene/configs/Feature_Extraction_Module_T5.yaml**

#### 3.1.2 训练

执行以下命令训练：

```bash
python train_feature_extraction.py --log_dir ../../excluded_dir/output_dir/logs/train_feature_extraction --output_dir ../../excluded_dir/output_dir/logs/train_feature_extraction/ckpt --resume_from_checkpoint '' --configs ../../configs/Feature_Extraction_Module_T5.yaml --device cpu --mixed_precision fp16
```

### 3.2 CogVideoX训练

#### 3.2.1 参数配置

详见**Mutiple_prompt_mutiple_scene/Script/finetune/schemas/args.py**

#### 3.2.2 训练

执行以下命令**训练**：

```bash
bash train_CogVideoX_ti2v.sh
```

## 4. inference

```bash
python inference.py \
--model_dir ../../excluded_dir/local_model/model_dir \
--multi_prompts 'Actor 1 is running on the playground.' 'Actor 2 is skiing on the snow.' 'Actor 2 is running on the playground.' \
--height 480 --width 720 --num_frames 49 --fps 15 \
--guidance_scale 6 --image_guidance_scale 6 \
--num_inference_steps 50 --device cuda --dtype fp16 --seed 42 \
--use_dynamic_cfg True --eta 0.0 \
--output_dir ../../excluded_dir/output_dir/inference/generate_video
```

