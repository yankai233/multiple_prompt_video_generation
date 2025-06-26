"""
1. 数据集
    <Video> -> <{V_i, P_i}>
2. 模型
    2.1 特征拼接
        P_i: [77, 512]

可以创新的点:
1. Classifier-free guidance: 与History Guidance相同:
    \epsilon(z_t,t,P+,{P+,I})=\epsilon(z_t,t,P-,{P+,I})+w1*(\epsilon(z_t,t,P+,{P+,I})-\epsilon(z_t,t,P-,{P+,I}))+
    w2*(\epsilon(z_t,t,P+,{P+,I})-\epsilon(z_t,t,P+,{None,None}))
    其中k随机去除一些{P,I}_k集合，让模型学习到即使不需要过去信息也能生成对应的图像
    TODO: k随机取
    现在代码已完成: k={1,2,...,i-1}


FIXME: train_CogVideoX的grad_norm为nan: 不使用H800
TODO: 数据集(完成部分)
1. TODO: 特征筛选模块
2. TODO: Classifier-free guidance: 随机遗忘{P_k,I_k}
3. TODO: 特征抽取模块随机掩码, 自监督
4. TODO:

"""

