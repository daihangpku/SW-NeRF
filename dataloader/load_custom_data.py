import os
import json
import numpy as np
import imageio
import cv2
import random
import torch

def pose_spherical(theta, phi, radius):
    trans_t = lambda t : torch.Tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ])

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi),np.cos(phi),0],
        [0,0,0,1],
    ])

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0,np.cos(th),0],
        [0,0,0,1],
    ])

    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]])) @ c2w
    return c2w

def load_custom_data(basedir, half_res=False, testskip=1):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
    
    frames = meta['frames']
    total_frames = len(frames)
    
    # 打乱帧的顺序以确保随机划分
    random.shuffle(frames)
    
    train_split = int(0.6 * total_frames)
    val_split = int(0.1 * total_frames)
    
    splits = {
        'train': frames[:train_split],
        'val': frames[train_split:train_split + val_split],
        'test': frames[train_split + val_split:]
    }
    
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in ['train', 'val', 'test']:
        imgs = []
        poses = []
        skip = testskip if s == 'test' else 1
        
        for frame in splits[s][::skip]:
            fname = os.path.join(basedir, frame['file_path'])
            img = imageio.imread(fname)
            if img.shape[-1] == 3:  # 如果是 RGB 图像
                img = np.concatenate([img, np.ones((*img.shape[:2], 1), dtype=img.dtype) * 255], axis=-1)  # 添加 Alpha 通道
            imgs.append(img)

            #imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal_x = meta['fl_x']
    focal_y = meta['fl_y']
    cx = meta['cx']
    cy = meta['cy']
    
    if half_res:
        H = H // 2
        W = W // 2
        focal_x = focal_x / 2.
        focal_y = focal_y / 2.
        cx = cx / 2.
        cy = cy / 2.
        
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    K = np.array([
        [focal_x, 0, cx],
        [0, focal_y, cy],
        [0, 0, 1]
    ])
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 360+1)[:-1]], 0)
    
    return imgs, poses, render_poses, K, [H, W, (focal_x + focal_y)*0.5], i_split