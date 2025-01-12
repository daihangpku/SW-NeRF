import trimesh
import numpy as np

def transform_mesh(input_obj_path, output_obj_path, scale: float, transform_matrix: np.ndarray):
    """
    读取 .obj 文件并按给定的缩放比例对 mesh 进行缩放。

    参数:
    input_obj_path: 输入的 .obj 文件路径
    output_obj_path: 输出的 .obj 文件路径
    scale: 缩放比例
    transform_matrix: 4x4 变换矩阵（包括平移、旋转、缩放等变换）
    
    返回:
    None
    """
    # 加载原始 mesh
    mesh = trimesh.load(input_obj_path)

    vertices = mesh.vertices  # [N, 3] 的顶点数组

    # 缩放
    vertices *= scale

    # 应用变换矩阵
    vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])  # [N, 4] 加上齐次坐标

    transformed_vertices = vertices_homogeneous.dot(transform_matrix.T)[:, :3]  # [N, 3]

    mesh.vertices = transformed_vertices

    mesh.export(output_obj_path)
    print(f"Transformed mesh saved to {output_obj_path}")

