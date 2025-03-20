import torch
import numpy as np
import matplotlib.pyplot as plt




def represent_state(pancakes_xy_pos, grid_size=(5, 5), x_min=0, x_max=1, y_min=0, y_max=1):
    """
    Transform the pancakes_xy_pos into a fixed-dimensional state representation.
    Inputs:
        pancakes_xy_pos: Tensor, shape (env, num_points, 2)
        grid_size: Grid size, e.g., (5, 5)
        x_min, x_max, y_min, y_max: Floats, range for normalizing the centroid
    Outputs:
        state_vectors: Tensor, shape (env, state_dim)
    """
    env, num_points, _ = pancakes_xy_pos.shape
    grid_x, grid_y = grid_size
    state_vectors = []

    for env_idx in range(env):
        # Get the points for the current environment
        points = pancakes_xy_pos[env_idx]  # shape (num_points, 2)

        # Initialize grid
        grid = torch.zeros(grid_x, grid_y)

        # Count points in the grid
        for x, y in points:
            i = int((y - y_min) / (y_max - y_min) * grid_y)
            j = int((x - x_min) / (x_max - x_min) * grid_x)
            if 0 <= i < grid_y and 0 <= j < grid_x:
                grid[i, j] += 1

        # Normalize to density
        if num_points > 0:
            grid_density = grid / num_points
        else:
            grid_density = grid

        # Calculate additional features
        total_points = num_points
        centroid = torch.mean(points, dim=0) if num_points > 0 else torch.tensor([0.5, 0.5])

        # Normalize the centroid
        centroid[0] = (centroid[0] - x_min) / (x_max - x_min)
        centroid[1] = (centroid[1] - y_min) / (y_max - y_min)

        # Concatenate feature vector
        grid_flat = grid_density.flatten()
        features = torch.cat([grid_flat, torch.tensor([total_points]), centroid])
        state_vectors.append(features)

    # Convert list to Tensor
    state_vectors = torch.stack(state_vectors)  # shape (env, state_dim)
    return state_vectors


# 绘制函数，根据represent_state的输出绘制栅格
def plot_state(points, state_vectors, grid_size, env_idx=0):
    state = state_vectors[env_idx]
    total_points = state[-3].item()
    centroid = state[-2:].numpy()

    grid_area = grid_size[0] * grid_size[1]
    grid_density = state[:grid_area].reshape(grid_size).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制点图
    ax_points = axes[0]
    ax_points.scatter(points[env_idx][:, 0], points[env_idx][:, 1], c='blue', s=50, alpha=0.6)
    ax_points.set_title(f"Environment {env_idx + 1}: Point Distribution")
    ax_points.set_xlim(0, 1)
    ax_points.set_ylim(0, 1)
    ax_points.set_aspect('equal', adjustable='box')

    # 绘制栅格图
    ax_grid = axes[1]
    im = ax_grid.imshow(grid_density, cmap='Blues', origin='lower', extent=[0, 1, 0, 1])
    ax_grid.set_title(f"Environment {env_idx + 1}: Grid and Centroid")
    plt.colorbar(im, ax=ax_grid)

    # 标注每个网格的计数
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = grid_density[i, j] * points[env_idx].shape[0]
            if count > 0:
                ax_grid.text(
                    (j + 0.5) / grid_size[1], (i + 0.5) / grid_size[0], 
                    int(count), color='black', ha='center', va='center'
                )

    # 绘制质心
    ax_grid.scatter([centroid[0]], [centroid[1]], color='red', marker='x')
    ax_grid.text(
        centroid[0], centroid[1], 
        f'Centroid: {np.round(centroid, 2)}', color='red', fontsize=12, ha='left', va='bottom'
    )

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# 示例使用
env_count = 4
points_per_env = 20
grid_size = (10, 10)

pancakes_xy_pos = torch.rand(env_count, points_per_env, 2)  # 4个环境，每个环境50个点
print(pancakes_xy_pos)
state_vectors = represent_state(pancakes_xy_pos, grid_size)

print("状态向量形状:", state_vectors.shape)
print(state_vectors)

# Plot the state for each environment
for env_idx in range(env_count):
    plot_state(pancakes_xy_pos, state_vectors, grid_size, env_idx)