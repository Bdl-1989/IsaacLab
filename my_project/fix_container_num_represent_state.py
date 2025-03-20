import torch

def encode_state(containers, max_containers=5, scene_bounds=[[0,2], [0,1]]):
    """
    输入:
        containers: List[Dict], 每个容器包含:
            "center": Tensor, 形状 (2,) 表示中心坐标 (x, y)
            "item_matrix": Tensor, 形状 (m, n) 表示 item 状态
    输出:
        state: Tensor, 形状 (max_containers * (2 + m*n),)
    """
    # 提取每个容器的特征
    features = []
    for container in containers:
        # 归一化中心坐标
        x_normalized = (container["center"][0] - scene_bounds[0][0]) / (scene_bounds[0][1] - scene_bounds[0][0])
        y_normalized = (container["center"][1] - scene_bounds[1][0]) / (scene_bounds[1][1] - scene_bounds[1][0])
        center = torch.tensor([x_normalized, y_normalized])          # 形状 (2,)
        item_matrix = container["item_matrix"] # 形状 (m, n)
        item_flatten = item_matrix.flatten()   # 形状 (m*n,)
        feature = torch.cat([center, item_flatten]) # 形状 (2 + m*n,)
        features.append(feature)
    
    # 填充或截断至 max_containers 个
    if len(features) < max_containers:
        padding = [torch.zeros_like(features[0])] * (max_containers - len(features))
        features += padding
    else:
        features = features[:max_containers]
    
    # 拼接所有特征
    state = torch.cat(features)  # 形状 (max_containers * (2 + m*n),)
    return state

# 示例使用
m, n = 2, 3
containers = [
    {
        "center": torch.tensor([0.5, 0.5]),
        "item_matrix": torch.randint(0, 2, (m, n)).float(),
    },
    {
        "center": torch.tensor([1.2, 0.5]),
        "item_matrix": torch.randint(0, 2, (m, n)).float(),
    }
]

state = encode_state(containers, max_containers=5)
print("状态向量形状:", state)
print(containers)