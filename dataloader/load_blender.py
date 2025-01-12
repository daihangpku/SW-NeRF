import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def visualize_cameras(poses):
    """可视化相机位姿和坐标轴方向
    Args:
        poses: [N, 4, 4] 相机到世界坐标系的变换矩阵
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 相机坐标轴的长度
    axis_length = 0.5
    
    for pose in poses:
        # 提取相机位置（变换矩阵的平移部分）
        pos = pose[:3, 3]
        
        # 提取相机方向（变换矩阵的旋转部分）
        R = pose[:3, :3]
        
        # 绘制相机位置
        ax.scatter(pos[0], pos[1], pos[2], c='black', marker='o')
        
        # 绘制相机坐标轴
        colors = ['r', 'g', 'b']  # x, y, z轴的颜色
        for i, color in enumerate(colors):
            direction = R[:, i]  # 第i个坐标轴的方向
            ax.quiver(pos[0], pos[1], pos[2],
                     direction[0], direction[1], direction[2],
                     length=axis_length, color=color)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置坐标轴比例相等
    ax.set_box_aspect([1,1,1])
    
    # 添加图例
    ax.plot([], [], [], 'r-', label='X axis')
    ax.plot([], [], [], 'g-', label='Y axis')
    ax.plot([], [], [], 'b-', label='Z axis')
    ax.legend()
    
    plt.title('Camera Poses Visualization')
    plt.show()
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        try:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
        except FileNotFoundError:
            metas[s] = None

    if all(meta is None for meta in metas.values()):
        # 如果没有找到任何分割文件，则自动划分数据集
        with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)
        frames = meta['frames']
        total_frames = len(frames)
        train_split = int(0.8 * total_frames)
        val_split = int(0.9 * total_frames)

        metas['train'] = {'frames': frames[:train_split]}
        metas['val'] = {'frames': frames[train_split:val_split]}
        metas['test'] = {'frames': frames[val_split:]}

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,360+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    #visualize_cameras(poses)
    return imgs, poses, render_poses, [H, W, focal], i_split


