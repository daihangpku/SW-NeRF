import trimesh
import os
import sys
import cv2
import cv2.aruco as aruco
from collections import Counter
from scipy.optimize import least_squares
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils import config_parser
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
def undistort_points(points, k1, k2, p1, p2):
    """
    对点进行去畸变
    """
    x = points[:, 0]
    y = points[:, 1]
    r2 = x*x + y*y
    
    # 径向畸变
    radial = (1 + k1*r2 + k2*r2*r2)
    
    # 切向畸变
    dx = 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    
    return np.column_stack([
        x*radial + dx,
        y*radial + dy
    ])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_and_measure_corners(corner_positions):
    """
    可视化角点位置并计算边长
    
    Args:
        corner_positions: ndarray, shape (4, 3), 4个角点的3D坐标
    Returns:
        float: 平均边长
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制角点
    ax.scatter(corner_positions[:, 0], corner_positions[:, 1], corner_positions[:, 2], 
              c='red', marker='o', s=100, label='Corners')
    
    # 连接角点形成方形
    # ArUco码的角点顺序是逆时针的
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edge_lengths = []
    
    for i, j in edges:
        # 计算边长
        edge_length = np.linalg.norm(corner_positions[i] - corner_positions[j])
        edge_lengths.append(edge_length)
        
        # 绘制边
        ax.plot([corner_positions[i, 0], corner_positions[j, 0]],
                [corner_positions[i, 1], corner_positions[j, 1]],
                [corner_positions[i, 2], corner_positions[j, 2]], 'b-')
    
    # 计算平均边长
    mean_edge_length = np.mean(edge_lengths)
    std_edge_length = np.std(edge_lengths)
    
    # 打印每条边的长度
    print("\nArUco码边长测量结果：")
    for i, length in enumerate(edge_lengths):
        print(f"边 {i+1}: {length:.3f} units")
    print(f"\n平均边长: {mean_edge_length:.3f} ± {std_edge_length:.3f} units")
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('ArUco码角点3D位置')
    
    # 使坐标轴等比例
    max_range = np.array([
        corner_positions[:, 0].max() - corner_positions[:, 0].min(),
        corner_positions[:, 1].max() - corner_positions[:, 1].min(),
        corner_positions[:, 2].max() - corner_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (corner_positions[:, 0].max() + corner_positions[:, 0].min()) * 0.5
    mid_y = (corner_positions[:, 1].max() + corner_positions[:, 1].min()) * 0.5
    mid_z = (corner_positions[:, 2].max() + corner_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 添加角点标签
    for i, corner in enumerate(corner_positions):
        ax.text(corner[0], corner[1], corner[2], f'C{i}', fontsize=12)
    
    plt.legend()
    plt.show()
    
    return mean_edge_length, edge_lengths



def get_ray_direction(corners, camera_params):
    """
    计算从相机到角点的射线方向
    """
    fl_x, fl_y, cx, cy, k1, k2, p1, p2, transform = camera_params
    
    # 将角点坐标转换为归一化坐标
    normalized_points = np.array([
        [(x - cx)/fl_x, (y - cy)/fl_y] 
        for x, y in corners[0]
    ])
    
    # 去畸变
    undistorted_points = undistort_points(normalized_points, k1, k2, p1, p2)
    
    # 转换为单位向量
    rays = np.column_stack([
        undistorted_points,
        np.ones(len(undistorted_points))
    ])
    rays = rays / np.linalg.norm(rays, axis=1)[:, np.newaxis]
    
    # 将射线转换到世界坐标系
    rotation = transform[:3, :3]
    rays = (rotation @ rays.T).T
    
    return rays

def triangulate_point(rays, camera_positions):
    """
    通过最小化射线到交点的距离来三角测量3D点位置
    """
    def distance_to_rays(point):
        point = point.reshape(3, 1)
        # 计算点到所有射线的距离
        distances = []
        for ray, pos in zip(rays, camera_positions):
            ray = ray.reshape(3, 1)
            pos = pos.reshape(3, 1)
            # 计算点到射线的垂直距离
            v = point - pos
            dist = np.linalg.norm(np.cross(v.ravel(), ray.ravel())) / np.linalg.norm(ray)
            distances.append(dist)
        return np.array(distances)

    # 使用射线的交点作为初始估计
    initial_point = np.mean(camera_positions, axis=0)
    
    # 优化求解
    result = least_squares(distance_to_rays, initial_point)
    return result.x

def calculate_3d_corners(frame_info, transform_data):
    """
    计算每个角点的3D位置
    """
    camera_params = []
    rays_list = []
    camera_positions = []
    
    for info in frame_info:
        frame = info['frame']
        transform = np.array(frame['transform_matrix'])
        
        params = (
            transform_data['fl_x'],
            transform_data['fl_y'],
            transform_data['cx'],
            transform_data['cy'],
            transform_data['k1'],
            transform_data['k2'],
            transform_data['p1'],
            transform_data['p2'],
            transform
        )
        
        # 计算相机位置（transform矩阵的最后一列）
        camera_pos = -transform[:3, :3].T @ transform[:3, 3]
        
        # 计算射线方向
        rays = get_ray_direction(info['corners'], params)
        
        camera_params.append(params)
        rays_list.append(rays)
        camera_positions.append(camera_pos)

    # 对每个角点进行三角测量
    corner_positions = []
    for i in range(4):  # ArUco码有4个角点
        corner_rays = [rays[i] for rays in rays_list]
        pos = triangulate_point(corner_rays, camera_positions)
        corner_positions.append(pos)
    
    return np.array(corner_positions)
def cal_scale(datapath, actual_size):
    # 读取 transform.json 文件
    transform_path = os.path.join(datapath, 'transforms.json')
    with open(transform_path, 'r') as f:
        transform_data = json.load(f)
    
    # 获取相机内参和外参
    fl_x = transform_data['fl_x']
    fl_y = transform_data['fl_y']
    cx = transform_data['cx']
    cy = transform_data['cy']
    k1 = transform_data['k1']
    k2 = transform_data['k2']
    p1 = transform_data['p1']
    p2 = transform_data['p2']
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_params)
    # 读取图片
    frames = transform_data['frames']
    all_corners = []
    all_ids = []
    frame_info = []
    for frame in frames:
        image_path = os.path.join(datapath, frame['file_path'].replace('images/', 'images_ori/'))
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at {image_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids.flatten())
            for corner, id in zip(corners, ids.flatten()):
                frame_info.append({'frame': frame, 'id': id, 'corners': corner})
    # ArUco 字典和检测器参数
        
    if all_ids:
        most_common_id = Counter(all_ids).most_common(1)[0][0]
        filtered_info = [info for info in frame_info if info['id'] == most_common_id]
    print(f"find ID: {most_common_id}, in total {len(filtered_info)} frames")
        # 保留其角点位置
        #for info in filtered_info:
            #print(f"Frame: {info['frame']['file_path']}, ArUco ID: {info['id']}, Corners: {info['corners']}")
    corner_positions = calculate_3d_corners(filtered_info, transform_data)
    print("ArUco码角点的3D位置：")
    for i, pos in enumerate(corner_positions):
        print(f"角点 {i}: {pos}")
    
    mean_length, edge_lengths = visualize_and_measure_corners(corner_positions)
    transform_matrix = calculate_transform_matrix(corner_positions)
    # 如果你知道ArUco码的实际尺寸，可以计算误差
    
    scale_error = actual_size / mean_length
    print(f"\n比例: {scale_error:.3%}")
    return scale_error, transform_matrix
    
def calculate_transform_matrix(corner_positions):
    """
    计算变换矩阵，使 ArUco 码的法向量与世界坐标的 z 轴正向对齐
    """
    # 计算 ArUco 码的法向量
    v1 = corner_positions[1] - corner_positions[0]
    v2 = corner_positions[2] - corner_positions[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # 计算旋转矩阵，使法向量与 z 轴对齐
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    s = np.linalg.norm(v)
    k = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_matrix = np.eye(3) + k + k.dot(k) * ((1 - c) / (s ** 2))
    
    # 构建 4x4 变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    
    return transform_matrix
def main():
    parser = config_parser()
    args = parser.parse_args()
    datadir = args.datadir
    basedir = args.basedir
    input_obj_path = os.path.join(args.basedir, args.expname, 'mesh.obj')
    output_obj_path = os.path.join(args.basedir, args.expname, 'transformed_mesh.obj')
    scale, transform_matrix = cal_scale(datadir,args.real_length)
    transform_mesh(input_obj_path, output_obj_path, scale, transform_matrix)
    

if __name__ == '__main__':
    main()

