import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
def get_rays(H, W, focal_or_K, c2w):
    """
    Generate rays in the world coordinate system.
    Args:
      H: int. Height of the image in pixels.
      W: int. Width of the image in pixels.
      focal_or_K: float or [3, 3] array. Focal length of the camera or camera intrinsic matrix.
      c2w: [4, 4] array. Camera-to-world transformation matrix.
    Returns:
      rays_o: [H, W, 3] Tensor. Ray origins in the world coordinate system.
      rays_d: [H, W, 3] Tensor. Ray directions in the world coordinate system.
    """
    # Create a meshgrid of pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    if isinstance(focal_or_K, float):
        # If focal_or_K is a float, treat it as the focal length
        focal = focal_or_K
        dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    else:
        # If focal_or_K is an array, treat it as the camera intrinsic matrix K
        K = focal_or_K
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d



def get_rays_np(H, W, focal_or_K, c2w):
    """
    Generate rays in the world coordinate system.
    Args:
      H: int. Height of the image in pixels.
      W: int. Width of the image in pixels.
      focal_or_K: float or [3, 3] array. Focal length of the camera or camera intrinsic matrix.
      c2w: [4, 4] array. Camera-to-world transformation matrix.
    Returns:
      rays_o: [H, W, 3] array. Ray origins in the world coordinate system.
      rays_d: [H, W, 3] array. Ray directions in the world coordinate system.
    """
    # Create a meshgrid of pixel coordinates
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    
    if isinstance(focal_or_K, float):
        # If focal_or_K is a float, treat it as the focal length
        focal = focal_or_K
        dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    else:
        # If focal_or_K is an array, treat it as the camera intrinsic matrix K
        K = focal_or_K
        dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling 
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Perform hierarchical sampling.
    Args:
      bins: [N, M] Tensor. Bin edges.
      weights: [N, M-1] Tensor. Weights for each bin.
      N_samples: int. Number of samples to draw.
      det: bool. If True, use deterministic sampling.
      pytest: bool. If True, use fixed random numbers for testing.
    Returns:
      samples: [N, N_samples] Tensor. Sampled points.
    """
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # [batch_size, N_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()# 确保 u 是连续的
    inds = torch.searchsorted(cdf, u, right=True) # 在cdf中查找u的位置
    below = torch.max(torch.zeros_like(inds-1), inds-1)#索引下边界
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # 收集 CDF 和 bin 值
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples#[batch_size, N_samples]