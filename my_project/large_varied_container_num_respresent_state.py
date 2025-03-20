import torch.nn as nn
import torch


class ContainerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)  # 输出形状 (batch, output_dim)
    

class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, state_dim=128):
        super().__init__()
        self.container_encoder = ContainerEncoder(input_dim, hidden_dim)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.output_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, features):
        """
        输入:
            features: Tensor, 形状 (num_containers, input_dim)
        输出:
            state: Tensor, 形状 (state_dim,)
        """
        encoded = self.container_encoder(features)  # 形状 (num_containers, hidden_dim)
        pooled = self.global_pool(encoded.unsqueeze(0))  # 形状 (1, hidden_dim, 1)
        pooled = pooled.squeeze(0).squeeze(-1)          # 形状 (hidden_dim,)
        state = self.output_layer(pooled)               # 形状 (state_dim,)
        return state
    


# 假设每个容器的输入维度是 2 + m*n = 2 + 9 = 11
encoder = StateEncoder(input_dim=11, state_dim=128)

# 生成动态数量的容器特征
num_containers = 3
features = torch.rand(num_containers, 11)  # 形状 (3, 11)
state = encoder(features)
print("状态向量形状:", state)  # 输出 (128,)



def update_container_positions(containers, v, dt, workarea_x_range=(0.0, 2.0)):
    """
    更新容器位置并检测进入 workarea 的容器。
    """
    valid_containers = []
    for container in containers:
        # 更新 x 坐标（假设传送带沿 x 轴移动）
        x_new = container["center"][0] - v * dt
        # 检测是否在 workarea 内
        if workarea_x_range[0] <= x_new <= workarea_x_range[1]:
            container["center"][0] = x_new
            valid_containers.append(container)
    return valid_containers

# 示例使用
v = 0.1  # 传送带速度
dt = 0.5
workarea_x_range = (0.0, 1.0)  # workarea 的 x 范围

# 初始容器位置（假设初始 x 坐标在右侧）
containers = [
    {"center": torch.tensor([2.0, 0.5]), "item_matrix": torch.zeros(3, 3)},
    {"center": torch.tensor([3.0, 0.5]), "item_matrix": torch.ones(3, 3)},
]

# 更新位置并筛选有效容器
valid_containers = update_container_positions(containers, v, dt, workarea_x_range)
print("有效容器数量:", len(valid_containers))