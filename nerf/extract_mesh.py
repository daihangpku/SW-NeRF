import numpy as np
from skimage import measure
import trimesh
import torch
from tqdm import tqdm
import os
def generate_viewdirs(num_views=100):
    """
    生成均匀分布在球面上的观察方向
    
    参数:
    num_views: 视角数量
    
    返回:
    array of shape (num_views, 3): 单位向量数组
    """
    indices = np.arange(0, num_views, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_views)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    return np.stack([x, y, z], axis=1)

def sample_grid(bounds, resolution, nerf_function, num_views=100, batch_size=1024):
    """
    对给定空间范围进行采样,生成密度场
    
    参数:
    bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    resolution: 每个维度的采样点数量
    nerf_function: 输入pos=[N,3], viewdir=[N,3]返回(r,g,b,rho)的函数
    num_views: 每个点采样的视角数量
    batch_size: 批处理大小
    """
    # 生成均匀分布的观察方向
    viewdirs = generate_viewdirs(num_views)  # [num_views, 3]
    
    # 生成采样点网格
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    z = np.linspace(bounds[2][0], bounds[2][1], resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 将所有点展平为 [N, 3] 数组
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # [N, 3]
    total_points = len(points)
    
    # 初始化结果数组
    avg_colors = np.zeros((total_points, 3))
    avg_densities = np.zeros(total_points)
    
    # 使用tqdm显示总体进度
    progress_bar = tqdm(total=total_points * num_views, desc="Sampling points")
    
    # 批量处理点和视角
    for start_idx in range(0, total_points, batch_size):
        end_idx = min(start_idx + batch_size, total_points)
        batch_points = points[start_idx:end_idx]  # [batch_size, 3]
        batch_colors = np.zeros((len(batch_points), num_views, 3))
        batch_densities = np.zeros((len(batch_points), num_views))
        # 对每个视角进行批量查询
        for view_idx, viewdir in enumerate(viewdirs):
            # 扩展视角方向以匹配批次中的点数
            batch_viewdirs = np.tile(viewdir[None], (len(batch_points), 1))  # [batch_size, 3]
            
            # 进行批量查询
            r, g, b, rho = nerf_function(batch_points, batch_viewdirs)
            
            # 存储结果
            batch_colors[:, view_idx] = np.stack([r, g, b], axis=-1)
            batch_densities[:, view_idx] = rho
            
            # 更新进度条
            progress_bar.update(len(batch_points))

        avg_colors[start_idx:end_idx] = np.mean(batch_colors, axis=1)
        avg_densities[start_idx:end_idx] = np.mean(batch_densities, axis=1)

    progress_bar.close()
    

    
    # 重塑回3D网格
    density_field = avg_densities.reshape(resolution, resolution, resolution)
    color_field = avg_colors.reshape(resolution, resolution, resolution, 3)
    
    return density_field, color_field, (X, Y, Z)

def generate_mesh(density_field, color_field, xyz_coords, density_threshold=0.5):
    """
    使用marching cubes算法从密度场生成mesh
    """
    # 使用marching cubes提取等值面
    verts, faces, normals, values = measure.marching_cubes(
        density_field, 
        level=density_threshold,
        spacing=(
            (xyz_coords[0][1,0,0] - xyz_coords[0][0,0,0]),
            (xyz_coords[1][0,1,0] - xyz_coords[1][0,0,0]),
            (xyz_coords[2][0,0,1] - xyz_coords[2][0,0,0])
        )
    )
    
    # 将顶点坐标变换到原始空间
    verts += np.array([
        xyz_coords[0][0,0,0],
        xyz_coords[1][0,0,0],
        xyz_coords[2][0,0,0]
    ])
    
    # 为每个顶点分配颜色
    vertex_colors = np.zeros((len(verts), 3))
    for i, vert in enumerate(verts):
        # 找到最近的采样点
        x_idx = np.argmin(np.abs(xyz_coords[0][:,0,0] - vert[0]))
        y_idx = np.argmin(np.abs(xyz_coords[1][0,:,0] - vert[1]))
        z_idx = np.argmin(np.abs(xyz_coords[2][0,0,:] - vert[2]))
        vertex_colors[i] = color_field[x_idx, y_idx, z_idx]
    
    # 创建mesh对象
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_normals=normals
    )
    
    return mesh

def nerf_to_mesh(nerf_function, bounds, resolution=64, density_threshold=8, num_views=100, batch_size=1024):
    """
    主函数：从NeRF模型生成mesh
    
    参数:
    nerf_function: 输入pos=[N,3], viewdir=[N,3]返回(r,g,b,rho)的函数
    bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    resolution: 采样分辨率
    density_threshold: 密度阈值
    num_views: 每个点采样的视角数量
    batch_size: 批处理大小
    """
    # 采样密度场和颜色场
    density_field, color_field, xyz_coords = sample_grid(
        bounds, resolution, nerf_function, num_views, batch_size
    )
    
    # 生成mesh
    mesh = generate_mesh(density_field, color_field, xyz_coords, density_threshold)
    
    return mesh

def main():
    from load_model import load_model
    bounds = [(-1., 0.9), (-1., 2.), (-4., -2.)]
    model, model_fine, optimizer, network_query_fn, args = load_model()
    
    # 获取模型所在的设备
    device = next(model_fine.parameters()).device
    
    # 包装查询函数以支持批处理
    def batch_query_fn(positions, viewdirs):
        # 确保输入是torch tensor并且在正确的设备上
        positions = torch.tensor(positions, dtype=torch.float32, device=device)  # [N, 3]
        viewdirs = torch.tensor(viewdirs, dtype=torch.float32, device=device)   # [N, 3]
        
        # 确保viewdirs是正确的维度
        if len(viewdirs.shape) == 3:  # 如果是[N, 1, 3]格式
            viewdirs = viewdirs.squeeze(1)  # 变成[N, 3]
        #print(positions.shape, viewdirs.shape)
        
        # 进行查询
        with torch.no_grad():
            outputs = network_query_fn(positions, viewdirs, model_fine)
        
        # 转换回numpy数组
        if len(outputs.shape) == 3:
            outputs = outputs.squeeze(1)
        rgb = outputs[..., :3].cpu().numpy()
        sigma = outputs[..., 3].cpu().numpy()
        
        return rgb[..., 0], rgb[..., 1], rgb[..., 2], sigma
    
    # 生成mesh
    mesh = nerf_to_mesh(batch_query_fn, bounds, resolution=args.resolution, batch_size=1024,density_threshold=args.threshold)
    savedir = os.path.join(args.basedir, args.expname, 'mesh.obj')
    mesh.export(savedir)
    print(f"Mesh saved to {savedir}")

if __name__ == "__main__":
    main()