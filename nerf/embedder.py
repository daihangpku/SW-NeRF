import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder:
   """位置编码器,将输入坐标映射到更高维的特征空间"""
   def __init__(self, **kwargs):
       self.kwargs = kwargs  # 存储配置参数
       self.create_embedding_fn()
       
   def create_embedding_fn(self):
       """创建编码函数列表"""
       embed_fns = []  # 存储所有编码函数
       d = self.kwargs['input_dims']  # 输入维度(通常是3,对应xyz坐标)
       out_dim = 0  # 输出维度
       
       # 是否保留原始输入
       if self.kwargs['include_input']:
           embed_fns.append(lambda x : x)  # 添加恒等函数
           out_dim += d
           
       max_freq = self.kwargs['max_freq_log2']  # 最大频率的log2值
       N_freqs = self.kwargs['num_freqs']  # 使用的频率数量
       
       # 计算频率序列:可以是对数采样或线性采样
       if self.kwargs['log_sampling']:
           freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  # 对数均匀采样
       else:
           freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)  # 线性均匀采样
           
       # 为每个频率创建正弦和余弦编码函数
       for freq in freq_bands:
           for p_fn in self.kwargs['periodic_fns']:  # 对每个周期函数(sin和cos)
               embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
               out_dim += d
                   
       self.embed_fns = embed_fns  # 保存所有编码函数
       self.out_dim = out_dim  # 保存输出维度
       
   def embed(self, inputs):
       """将输入通过所有编码函数,并在最后一个维度拼接"""
       return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
   
def get_embedder(multires, i=0):
   """
   创建位置编码器的工厂函数
   Args:
       multires: 使用的频率数量
       i: 如果为-1,返回恒等变换
   Returns:
       embed: 编码函数
       out_dim: 输出维度
   """
   if i == -1:
       return nn.Identity(), 3
   
   # 编码器的配置参数
   embed_kwargs = {
               'include_input' : True,    # 是否保留原始输入
               'input_dims' : 3,          # 输入维度(xyz坐标)
               'max_freq_log2' : multires-1,  # 最大频率的log2值
               'num_freqs' : multires,    # 频率数量 
               'log_sampling' : True,      # 使用对数采样
               'periodic_fns' : [torch.sin, torch.cos],  # 使用sin和cos作为周期函数
   }
   
   embedder_obj = Embedder(**embed_kwargs)
   embed = lambda x, eo=embedder_obj : eo.embed(x)  # 创建编码函数
   return embed, embedder_obj.out_dim  # 返回编码函数和输出维度