import os 
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

# 定义平移矩阵，平移量是 t
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 定义绕某一角度 (phi) 绕 Y 轴旋转的旋转矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 定义绕某一角度 (theta) 绕 X 轴旋转的旋转矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

# 将 Rodrigues 旋转公式转化为旋转向量
def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)  # 获取矩阵的迹
  trc2 = (trc - 1.) / 2.  # 计算迹的平方
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # 计算旋转轴
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)  # 计算旋转角度
    tHetaf = tHeta / (2 * (np.sin(tHeta)))  # 归一化旋转角度
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s  # 计算旋转向量
  return omega

# 将旋转向量转换为 Rodrigues 旋转矩阵
def rodrigues_rot_to_mat(r):
  wx, wy, wz = r  # 旋转向量
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)  # 计算旋转角度
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])  # 初始化旋转矩阵
  # 构造旋转矩阵
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R

# 生成球形坐标系的旋转矩阵
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)  # 计算平移矩阵
    c2w = rot_phi(phi/180.*np.pi) @ c2w  # 绕phi角度旋转
    c2w = rot_theta(theta/180.*np.pi) @ c2w  # 绕theta角度旋转
    # 将坐标系反转，使得Z轴朝上
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

# 加载Blender数据集，包含训练、验证和测试数据
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        # 读取转换数据文件
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]

        imgs = []
        poses = []
        times = []
        skip = testskip  # 跳过帧数，用于测试集数据处理
            
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')  # 图像路径
            imgs.append(imageio.imread(fname))  # 读取图像
            poses.append(np.array(frame['transform_matrix']))  # 读取位姿矩阵
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)  # 获取时间
            times.append(cur_time)

        assert times[0] == 0, "Time must start at 0"  # 确保时间从0开始

        # 归一化处理图像并转换数据类型
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # 合并所有数据集的图像、位姿和时间
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    H, W = imgs[0].shape[:2]  # 获取图像的高度和宽度
    camera_angle_x = float(meta['camera_angle_x'])  # 获取相机的角度
    focal = .5 * W / np.tan(.5 * camera_angle_x)  # 计算焦距

    # 如果渲染数据存在，则读取渲染数据
    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))  # 渲染的位姿矩阵
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        # 如果没有渲染数据，则生成一定角度范围内的位姿矩阵
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    render_times = torch.linspace(0., 1., render_poses.shape[0])  # 渲染的时间从0到1

    # 如果需要半分辨率图像，则进行图像缩小处理
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))  # 半分辨率图像
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    print(f"imgs.shape at loading: {imgs.shape}")
    # 返回图像、位姿、时间、渲染位姿、渲染时间以及相机参数
    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split
