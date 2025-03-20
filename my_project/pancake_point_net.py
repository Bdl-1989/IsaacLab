import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    def __init__(self, feature_dim=64, x_min=0, x_max=1, y_min=0, y_max=1):
        super(PointNetEncoder, self).__init__()
        # 共享 MLP
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),  # 输入是 2D 坐标 (x, y)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # 最大池化
        self.pool = nn.AdaptiveMaxPool1d(1)
        # 输出层
        self.output_layer = nn.Linear(64, feature_dim)

    def forward(self, x):
        """
        输入:
        x: Tensor, 形状为 (num_points, 2)
        输出:
        features: Tensor, 形状为 (feature_dim,)
        """
        num_points, _ = x.shape

        # Min-Max 归一化
        x[:, 0] = (x[:, 0] - self.x_min) / (self.x_max - self.x_min)
        x[:, 1] = (x[:, 1] - self.y_min) / (self.y_max - self.y_min)
        print(x)
        x = x.view(-1, 2)  # 展平为 (num_points, 2)
        x = self.mlp(x)  # 形状为 (num_points, 64)
        x = x.view(1, num_points, -1)  # 增加批次维度，形状为 (1, num_points, 64)
        x = x.permute(0, 2, 1)  # 形状为 (1, 64, num_points)
        x = self.pool(x)  # 形状为 (1, 64, 1)
        x = x.squeeze(-1)  # 形状为 (1, 64)
        x = self.output_layer(x)  # 形状为 (1, feature_dim)
        x = x.squeeze(0)  # 移除批次维度，形状为 (feature_dim,)

        return x

# # 示例使用
# points = torch.rand(4, 50, 2)  # 4个环境，每个环境50个点
# encoder = PointNetEncoder(feature_dim=64)
# state_vectors = encoder(points)
# print("PointNet 特征形状:", state_vectors.shape)
# print(state_vectors)

 
# import matplotlib.pyplot as plt

 
# # Select one of the batches, for example the first batch
# batch_idx = 0
# points_to_plot = points[batch_idx].numpy()

# # Plot the points
# plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1])
# plt.title("2D Points for Batch {}".format(batch_idx))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.show()


# 创建模型
# 创建一个形状为 (num_points, 2) 的输入
num_points = 100
x = torch.randn(num_points, 2)

# 初始化模型
encoder = PointNetEncoder(feature_dim=64)

# 前向传播
features = encoder(x)
print(features.shape)  # 输出: torch.Size([64])