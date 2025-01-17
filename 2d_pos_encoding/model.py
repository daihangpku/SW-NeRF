import torch.nn as nn
class Model(nn.Module):
    '''
    define model you desire and rename the class
    '''
    def __init__(self, input_dimension: int, layer_num: int, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        layers = []
        current_dim = input_dimension
        
        # 构建隐藏层
        for i in range(layer_num):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用Xavier均匀分布初始化权重"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        前向传播
        参数:
        - x (torch.Tensor): 输入张量,形状为 (batch_size, input_dimension)
        返回:
        - torch.Tensor: 输出张量,形状为 (batch_size, output_dim)
        """
        return self.model(x)
    

