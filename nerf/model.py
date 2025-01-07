import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NeRF网络模型定义
class NeRF(nn.Module):
   def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
       """ 
       参数:
           D: 网络深度
           W: 网络宽度(隐藏层维度)
           input_ch: 位置编码后的输入通道数
           input_ch_views: 视角方向编码后的输入通道数
           output_ch: 输出通道数(RGB + density)
           skips: 跳跃连接的层索引
           use_viewdirs: 是否使用视角方向信息
       """
       super(NeRF, self).__init__()
       self.D = D
       self.W = W
       self.input_ch = input_ch
       self.input_ch_views = input_ch_views
       self.skips = skips
       self.use_viewdirs = use_viewdirs
       
       # 处理空间位置的MLP网络
       self.pts_linears = nn.ModuleList(
           [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
       
       # 处理视角方向的MLP网络(官方实现)
       self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

       # 论文中的实现方式
       # self.views_linears = nn.ModuleList(
       #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
       
       if use_viewdirs:
           # 使用视角方向时的输出层
           self.feature_linear = nn.Linear(W, W)  # 特征层
           self.alpha_linear = nn.Linear(W, 1)    # 密度预测层
           self.rgb_linear = nn.Linear(W//2, 3)   # RGB颜色预测层
       else:
           # 不使用视角方向时的输出层
           self.output_linear = nn.Linear(W, output_ch)

   def forward(self, x):
       # 将输入分割为位置编码和视角编码
       input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
       h = input_pts
       
       # 通过位置MLP网络
       for i, l in enumerate(self.pts_linears):
           h = self.pts_linears[i](h)
           h = F.relu(h)
           if i in self.skips:
               h = torch.cat([input_pts, h], -1)  # 跳跃连接

       if self.use_viewdirs:
           # 使用视角信息的处理流程
           alpha = self.alpha_linear(h)           # 预测密度
           feature = self.feature_linear(h)       # 提取特征
           h = torch.cat([feature, input_views], -1)  # 特征与视角信息拼接
       
           # 通过视角MLP网络
           for i, l in enumerate(self.views_linears):
               h = self.views_linears[i](h)
               h = F.relu(h)

           rgb = self.rgb_linear(h)              # 预测RGB颜色
           outputs = torch.cat([rgb, alpha], -1)  # 组合最终输出
       else:
           # 不使用视角信息时直接输出
           outputs = self.output_linear(h)

       return outputs    

   def load_weights_from_keras(self, weights):
       """从Keras模型加载预训练权重"""
       assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
       
       # 加载位置MLP的权重
       for i in range(self.D):
           idx_pts_linears = 2 * i
           self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
           self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
       
       # 加载特征层权重
       idx_feature_linear = 2 * self.D
       self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
       self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

       # 加载视角MLP权重
       idx_views_linears = 2 * self.D + 2
       self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
       self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

       # 加载RGB输出层权重
       idx_rbg_linear = 2 * self.D + 4
       self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
       self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

       # 加载密度输出层权重
       idx_alpha_linear = 2 * self.D + 6
       self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
       self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))