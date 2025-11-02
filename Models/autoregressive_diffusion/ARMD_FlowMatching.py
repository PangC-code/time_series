import math
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

# 假设 Linear 模型和其他工具函数已定义
# from Models.autoregressive_diffusion.linear import Linear
from Models.autoregressive_diffusion.model_utils import default, identity, extract
from Models.autoregressive_diffusion.vector_field_net import VectorFieldNet ,TransformerVectorFieldNet

class ARMD_FlowMatching(nn.Module):
    """
    Auto-Regressive Moving Diffusion (ARMD) model refactored with Flow Matching.
    """
    def __init__(
            self,
            seq_length,      # 预测/历史长度
            feature_size,
            # 可以保留或添加与模型结构相关的参数
            total_timesteps=96, # 用于将 t \in [0,1] 映射到整数
            # w_grad=True,
            
            # 普通向量场
            # hidden_dim=256,
            # time_emb_dim=32,

            # 
            d_model=128,      # 使用传入的 d_model
            n_head=4,        # 使用传入的 n_head
            n_layers=2,    # 使用传入的 n_layers
            time_emb_dim=32,      # time_emb_dim 也可以做成可配置的

            sigma_min=1e-4,
            **kwargs
    ):
        super().__init__()

        # self.sigma_min = sigma_min
        # self.seq_length = int(seq_length)
        # self.feature_size = feature_size
        # self.total_timesteps = total_timesteps

        self.seq_length = int(seq_length)
        self.feature_size = int(feature_size)
        self.total_timesteps = int(total_timesteps)
        self.sigma_min = float(sigma_min)

        # 这个网络现在的作用是学习向量场 v(x_t, t)
        # self.vector_field_network = Linear(n_feat=feature_size, n_channel=seq_length, w_grad=w_grad, **kwargs)
        # self.vector_field_network = VectorFieldNet(
        #     n_feat=self.feature_size,
        #     n_channel=self.seq_length,
        #     hidden_dim=self.hidden_dim,
        #     time_emb_dim=self.time_emb_dim
        # )

        self.vector_field_network = TransformerVectorFieldNet(
            n_feat=self.feature_size,
            n_channel=self.seq_length,
            d_model=int(d_model),      # 使用传入的参数，并确保是整数
            n_head=int(n_head),        # 使用传入的参数，并确保是整数
            n_layers=int(n_layers),    # 使用传入的参数，并确保是整数
            time_emb_dim=int(time_emb_dim)) # 同
        self.loss_fn = F.mse_loss
         
    def forward(self, x, **kwargs):
        """
        训练步骤。
        x 的形状是 (batch, history_len + pred_len, feature_size)
        """
        # 1. Prepare data path endpoints
        history_len = self.seq_length
        x_history = x[:, :history_len, :]  # Path endpoint x_1
        x_future = x[:, history_len:, :]    # Path startpoint x_0
        x_0 = x_future
        x_1 = x_history

        b = x_0.shape[0]
        device = x.device

        # 2. Sample random time t in [0, 1]
        t = torch.rand(b, 1, 1, device=device)

        # 3. Construct the Conditional Flow Matching (CFM) path and target
        one_minus_sigma_min = 1 - self.sigma_min
        x_t = (1 - one_minus_sigma_min * t) * x_0 + one_minus_sigma_min * t * x_1
        target_vector_field = one_minus_sigma_min * (x_1 - x_0)

        # 4. Predict the vector field using the network
        # Map t from [0, 1] to integer timesteps [0, total_timesteps - 1]
        # Ensure the shape is (b,) for indexing
        time_steps = t.squeeze() * self.total_timesteps
        time_steps = torch.clamp(time_steps, 0, self.total_timesteps - 1).long()

        predicted_vector_field = self.vector_field_network(x_t, time_steps)

        # 5. Calculate the loss
        loss = self.loss_fn(predicted_vector_field, target_vector_field)
        return loss

    @torch.no_grad()
    def generate_mts(self, x_history, steps=100):
        """
        推理/采样步骤。
        从历史数据 x_history (x_1) 开始，通过ODE求解器生成未来数据 x_0。
        """
        device = x_history.device
        b = x_history.shape[0]
        
        # 初始状态是历史序列，对应 t=1
        x_t = x_history.clone()
        
        # 时间步长
        dt = 1.0 / steps
        
        # 从 t=1 到 t=0 进行逆向积分 (使用欧拉法)
        time_range = torch.linspace(1.0, 0.0, steps + 1)
        for i in tqdm(range(steps), desc="Flow Matching Sampling"):
            t_val = time_range[i]
            t = torch.full((b,), t_val * self.total_timesteps, device=device, dtype=torch.long)
            t = torch.clamp(t, 0, self.total_timesteps - 1).long()  #<----
            # 预测当前位置的向量场 v(x_t, t)
            pred_v = self.vector_field_network(x_t, t)
            
            # 欧拉法更新: x_{t-dt} = x_t - v * dt
            x_t = x_t - pred_v * dt
        
        # 返回最终在 t=0 时的状态，即预测的未来序列
        return x_t