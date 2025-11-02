import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.autoregressive_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp, RevIN, SinusoidalPosEmb, extract                                                       
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

timesteps = 96

class Linearv1(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        w_grad=True,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(n_channel, n_channel)
        self.betas = linear_beta_schedule(96)
        self.betas_dev = cosine_beta_schedule(96)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_dev = 1. - self.betas_dev
        self.w = torch.nn.Parameter(torch.FloatTensor(self.alphas_cumprod.numpy()), requires_grad=w_grad)
        self.w_dev = torch.nn.Parameter(torch.FloatTensor(self.alphas_dev.numpy()), requires_grad=False)

    def forward(self, input_, t, training=True):
        noise = torch.randn_like(input_)
        if not training:
            noise=0
        input_+= self.w_dev[t[0]]*noise
        x_tmp = self.linear(input_.permute(0,2,1)).permute(0,2,1)
        alpha = self.w[t[0]]
        output = (alpha*input_ + (1-2*alpha)*x_tmp) / (1-1*alpha)**(1/2)
        #if not training:
            #print('alpha:',alpha)
            #print('para:',1-1*alpha)
            #print('dis:',x_tmp.mean())
            #print('loss:',((1-1*alpha)*x_tmp).mean())

        output = output.to(torch.float32)

        return output

class Linear(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        w_grad=True,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(n_channel, n_channel)

        # --- 开始修改 ---
        betas = linear_beta_schedule(96)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        betas_dev = cosine_beta_schedule(96)
        alphas_dev = 1. - betas_dev
        
        # 使用 register_buffer，这样它们会自动移动到正确的设备
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_dev', alphas_dev.float())
        
        # 将 w 定义为一个可学习的参数，长度为96
        # 它不再直接由 alphas_cumprod 初始化，但我们可以用其初始化
        self.w = nn.Parameter(alphas_cumprod.clone().float(), requires_grad=w_grad)
        # --- 结束修改 ---
        self.time_mlp = nn.Sequential(
              SinusoidalPosEmb(time_emb_dim),
              nn.Linear(time_emb_dim, time_emb_dim * 4),
              nn.Mish(),
              nn.Linear(time_emb_dim * 4, time_emb_dim)
          )

    # def forward(self, input_, t, training=True):
    #     # --- 开始修改 ---
    #     # w_dev 现在不再是 nn.Parameter，而是 buffer
    #     # 并且我们应该为批次中的每个样本使用其对应的时间步
    #     w_dev_t = self.alphas_dev[t] # 使用整个 t 张量进行索引
    #     w_dev_t = w_dev_t.unsqueeze(-1).unsqueeze(-1) # 扩展维度以匹配 input_

    #     noise = torch.randn_like(input_) if training else 0.
        
    #     # 为每个样本添加其对应时间步的噪声
    #     input_ = input_ + w_dev_t * noise 
        
    #     x_tmp = self.linear(input_.permute(0,2,1)).permute(0,2,1)
        
    #     # alpha 也应该为批次中的每个样本分别计算
    #     alpha = self.w[t] # 使用整个 t 张量进行索引
    #     alpha = alpha.unsqueeze(-1).unsqueeze(-1) # 扩展维度以匹配 input_

    #     # --- 结束修改 ---
        
    #     # 下面的计算现在是元素级别的，适用于整个批次
    #     output = (alpha*input_ + (1-2*alpha)*x_tmp) / torch.sqrt(1-1*alpha)
        
    #     output = output.to(torch.float32)
    #     return output
    def forward(self, input_, t, training=True):
        # 仍然可以保留时间编码，但让它以更简单的方式影响计算
        # 例如，使用 FiLM (Feature-wise Linear Modulation)
        t_emb = self.time_mlp(t) # 假设你添加了一个 time_mlp
        gamma = self.to_gamma(t_emb) # 线性层
        beta = self.to_beta(t_emb)   # 线性层
        
        # 将 gamma 和 beta 扩展维度
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # FiLM-like modulation
        x_tmp = self.linear(input_.permute(0,2,1)).permute(0,2,1)
        output = x_tmp * gamma + beta
        return output