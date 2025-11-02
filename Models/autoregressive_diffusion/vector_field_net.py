import torch
from torch import nn
import torch.nn.functional as F
from .model_utils import SinusoidalPosEmb#, Mish
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class VectorFieldNet(nn.Module):
    """
    一个基于MLP的简单向量场网络。
    它将时间序列展平，与时间编码拼接，然后通过一个MLP来预测向量场。
    """
    def __init__(
        self,
        n_feat,          # 每个时间步的特征数 (例如 7)
        n_channel,       # 时间序列的长度 (例如 96)
        hidden_dim=256,  # MLP中间层的维度
        time_emb_dim=32, # 时间编码的维度
        **kwargs         # 接收并忽略其他不需要的参数
    ):
        super().__init__()
        self.seq_len = n_channel
        self.feature_size = n_feat

        # 1. 时间编码网络
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Mish(), # 使用Mish激活函数，效果通常比ReLU好
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 2. 主网络 (MLP)
        input_dim = self.seq_len * self.feature_size
        
        self.main_net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.SiLU(), # SiLU (或Swish) 也是一个很好的激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim) # 输出维度和输入维度相同
        )

    def forward(self, x_t, t):
        """
        - x_t: 中间状态的时间序列, shape: (batch, seq_len, feature_size)
        - t: 时间步, shape: (batch,)
        """
        
        # (1) 获取时间编码
        # SinusoidalPosEmb 需要整数时间步，所以确保 t 是 long 类型
        t_emb = self.time_mlp(t.long()) 
        
        # (2) 展平时间序列
        # shape: (batch, seq_len * feature_size)
        x_flat = x_t.reshape(x_t.shape[0], -1)
        
        # (3) 拼接输入和时间编码
        # shape: (batch, seq_len * feature_size + time_emb_dim)
        net_input = torch.cat([x_flat, t_emb], dim=1)
        
        # (4) 通过主网络得到输出
        output_flat = self.main_net(net_input)
        
        # (5) 将输出 reshape 回时间序列的形状
        # shape: (batch, seq_len, feature_size)
        return output_flat.reshape(x_t.shape)

# 在 vector_field_net.py 中
# import torch
# from torch import nn
# from .model_utils import SinusoidalPosEmb, Mish

class TransformerVectorFieldNet(nn.Module):
    def __init__(self, n_feat, n_channel, d_model=128, n_head=4, n_layers=2, time_emb_dim=32, **kwargs):
        super().__init__()
        self.seq_len = n_channel
        self.feature_size = n_feat

        # 1. 时间编码网络 (不变)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Mish(), # 使用Mish激活函数，效果通常比ReLU好
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 2. 输入投影层
        # 将每个时间步的特征从 n_feat 投影到 d_model
        self.input_proj = nn.Linear(n_feat, d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. 时间编码融合
        # 用于将时间编码 t_emb 融入到序列中
        self.time_proj = nn.Linear(time_emb_dim, d_model)

        # 5. 输出投影层
        # 将每个时间步的特征从 d_model 投影回 n_feat
        self.output_proj = nn.Linear(d_model, n_feat)
        
    def forward(self, x_t, t):
        # x_t shape: (batch, seq_len, n_feat)
        # t shape: (batch,)
        
        t_emb = self.time_mlp(t.long())
        
        # (1) 投影输入
        x_proj = self.input_proj(x_t) # shape: (batch, seq_len, d_model)
        
        # (2) 将时间编码融入序列
        # 将t_emb扩展并加到序列的每个位置上，这是一种常见的融合方式
        t_proj = self.time_proj(t_emb).unsqueeze(1) # shape: (batch, 1, d_model)
        seq_with_time = x_proj + t_proj
        
        # (3) 通过 Transformer Encoder
        transformer_output = self.transformer_encoder(seq_with_time) # shape: (batch, seq_len, d_model)
        
        # (4) 投影回原始特征维度
        output = self.output_proj(transformer_output) # shape: (batch, seq_len, n_feat)
        
        return output