import os
import torch
import numpy as np
import random
import argparse

import warnings
warnings.filterwarnings("ignore")

from engine.solver import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader, build_dataloader_cond

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Parameters:
    - seed (int): The seed value.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional steps for CuDNN backend
    os.environ['PYTHONHASHSEED'] = str(seed)

# Example usage:
set_seed(2023)

#class Args_Example:
#    def __init__(self) -> None:
#        self.config_path = './Config/etth.yaml'
#        self.save_dir = './forecasting_exp'
#        self.gpu = 0
#        os.makedirs(self.save_dir, exist_ok=True)

class Args_Example:
    def __init__(self, config_path, save_dir, gpu):
        self.config_path = config_path
        self.save_dir = save_dir
        self.gpu = gpu
        os.makedirs(self.save_dir, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration and directories.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp',
                        help='Directory to save experiment results.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #args =  Args_Example()
    args_parsed = parse_arguments()
    args = Args_Example(args_parsed.config_path, args_parsed.save_dir, args_parsed.gpu)
    seq_len = 192
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = instantiate_from_config(configs['model']).to(device)
    #model.use_ff = False
    model.fast_sampling = True
    #configs['solver']['max_epochs']=100
    dataloader_info = build_dataloader(configs, args)
    dataloader = dataloader_info['dataloader']
    trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader':dataloader})
    trainer.train()
    args.mode = 'predict'
    args.pred_len = seq_len
    test_dataloader_info = build_dataloader_cond(configs, args)
    test_scaled = test_dataloader_info['dataset'].samples
    scaler = test_dataloader_info['dataset'].scaler
    seq_length, feat_num = seq_len*2, test_scaled.shape[-1]
    pred_length = seq_len
    real = test_scaled
    test_dataset = test_dataloader_info['dataset']
    test_dataloader = test_dataloader_info['dataloader']
    sample, real_ = trainer.sample_forecast(test_dataloader, shape=[seq_len, feat_num])

    print("\n" + "="*50)
print("Performing Sanity Check with RANDOM input history...")
print("="*50)

# 从测试集中取一个批次的数据，我们只需要它的形状和真实目标
try:
    first_batch = next(iter(test_dataloader))
    if isinstance(first_batch, list): # 处理 dataloader 可能返回列表的情况
        first_batch = first_batch[0]
except StopIteration:
    print("Test dataloader is empty. Cannot perform sanity check.")
    first_batch = None

if first_batch is not None:
    # 准备一个批次的真实数据，用于对比
    real_data_for_check = first_batch.to(device)
    real_history_for_check = real_data_for_check[:, :seq_len, :]
    real_future_for_check = real_data_for_check[:, seq_len:, :]

    # --- 关键步骤：创建完全随机的历史数据 ---
    # 使用和真实历史数据相同的形状和设备
    random_history = torch.randn_like(real_history_for_check)
    
    print(f"Using a random history tensor of shape: {random_history.shape}")

    # 使用 Trainer 的一个临时实例或直接调用模型来进行预测
    # 为了简单，我们直接调用模型。注意要使用 self.ema.ema_model
    trainer.model.eval() # 确保模型在评估模式
    with torch.no_grad():
        sampling_steps = configs.get('forecasting', {}).get('sampling_steps', 100)
        # 使用 ema.ema_model 进行预测，这和 Trainer 内部的做法一致
        predicted_future_from_random = trainer.ema.ema_model.generate_mts(random_history, steps=sampling_steps)

    # 将预测结果和真实未来值转换为 numpy 用于计算指标
    sample_np = predicted_future_from_random.cpu().numpy()
    real_np = real_future_for_check.cpu().numpy()

    # 计算 MSE 和 MAE
    mse_random = mean_squared_error(sample_np.flatten(), real_np.flatten())
    mae_random = mean_absolute_error(sample_np.flatten(), real_np.flatten())

    print("\nResults from RANDOM input history:")
    print(f"MSE: {mse_random}, MAE: {mae_random}")
    print("="*50)

    if mse_random < 0.5: # 设定一个阈值，比如0.5
        print("\nWARNING: The MSE from random input is suspiciously low.")
        print("This strongly suggests a DATA LEAKAGE issue in your pipeline.")
    else:
        print("\nINFO: The MSE from random input is high, as expected.")
        print("This suggests that the model is genuinely relying on the input history.")

    # mask = test_dataset.masking #-------------------
    mse = mean_squared_error(sample.reshape(-1), real_.reshape(-1))
    mae = mean_absolute_error(sample.reshape(-1), real_.reshape(-1))
    print(mse,mae)

