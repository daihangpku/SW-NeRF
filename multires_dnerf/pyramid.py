import torch
import sys
import os
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dataloader.load_blender
def create_gaussian_kernel(kernel_size, sigma,channels=3):
    """
    创建一个高斯核
    :param kernel_size: 高斯核的大小（奇数）
    :param sigma: 高斯核的标准差
    :return: 高斯核
    """
    # 计算高斯核
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    grid = torch.meshgrid(coords, coords)
    kernel = torch.exp(-(grid[0]**2 + grid[1]**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # 扩展为卷积核的形式 (1, 1, kernel_size, kernel_size)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def apply_gaussian_blur(images, kernel_size=5, sigma=1.0):
    """
    使用高斯模糊对图像进行模糊
    :param images: 输入图像 (N, C, H, W)
    :param kernel_size: 高斯核的大小
    :param sigma: 高斯核的标准差
    :return: 模糊后的图像
    """
    # 创建高斯核
    kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # 将高斯核放到 GPU 上，如果有的话
    kernel = kernel.to(images.device)
    
    # 使用 F.conv2d 进行模糊处理
    # 在输入图像上应用高斯核，padding=kernel_size//2 确保图像大小不变
    blurred_images = F.conv2d(images, kernel, padding=kernel_size // 2, groups=images.shape[1])
    
    return blurred_images

def generate_laplacian_pyramid_batch(images, levels=4, kernel_size=3, sigma=1.0):
    """
    生成图像的拉普拉斯金字塔（批量化处理），下采样时使用高斯模糊
    :param images: 输入图像批次，形状为 (N, H, W, 3)
    :param levels: 金字塔的层数
    :param kernel_size: 高斯核的大小
    :param sigma: 高斯核的标准差
    :return: 拉普拉斯金字塔列表，每个元素是一个形状为 (N, Hi, Wi, 3) 的张量
    """
    # 将输入的图像从 (N, H, W, 3) 转换为 (N, 3, H, W)
    images = images.permute(0, 3, 1, 2)
    
    # 初始化高斯金字塔和拉普拉斯金字塔
    gaussian_pyramid = [images]
    laplacian_pyramid = []
    
    # 生成高斯金字塔
    for i in range(levels):
        # 在下采样前先进行高斯模糊
        blurred = apply_gaussian_blur(gaussian_pyramid[i], kernel_size, sigma)
        # 使用 F.interpolate 进行下采样
        downsampled = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
        gaussian_pyramid.append(downsampled)
    
    # 生成拉普拉斯金字塔
    for i in range(levels-1):
        # 对每一层的高斯金字塔进行上采样
        upsampled = F.interpolate(gaussian_pyramid[i+1], size=gaussian_pyramid[i].shape[2:], mode='bilinear', align_corners=False)
        # 计算拉普拉斯金字塔（差异部分）
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian.permute(0, 2, 3, 1))  
    
    # 最后一层的拉普拉斯金字塔就是最后的高斯金字塔层
    laplacian_pyramid.append(gaussian_pyramid[-2].permute(0, 2, 3, 1)) 
    return laplacian_pyramid

def reconstruct_image_from_pyramid_batch(laplacian_pyramid):
    """
    从拉普拉斯金字塔恢复图像（批量化处理）
    :param laplacian_pyramid: 拉普拉斯金字塔列表，形状为 (1, 3, Hi, Wi)
    :return: 恢复的原始图像，形状为 (H, W, 3)
    """
    # 从底层开始逐层恢复图像
    laplacian_pyramid = [l.permute(0, 3, 1, 2) for l in laplacian_pyramid]
    reconstructed_images = laplacian_pyramid[-1]
    
    for i in range(len(laplacian_pyramid)-2, -1, -1):
        # 对当前层的图像进行上采样
        upsampled = F.interpolate(reconstructed_images,size=laplacian_pyramid[i].shape[2:], mode='bilinear', align_corners=False)
        # 恢复当前层图像
        reconstructed_images = upsampled + laplacian_pyramid[i]
    
    return reconstructed_images.permute(0, 2, 3, 1)  # 转回 (N, H, W, 3)
from PIL import Image
import numpy as np
def save_tensors_as_images(tensor_list, output_dir, prefix='image', use_pil=True):
    """
    将一个包含多个 N × H × W × 3 张量的列表中的每个 H × W × 3 图像保存到指定的文件夹中。

    参数：
        tensor_list (list of torch.Tensor): 输入的张量列表，每个张量形状为 (N, H, W, 3)。
        output_dir (str): 图像保存的目标文件夹路径。
        prefix (str): 保存的图像文件名前缀（默认：'image'）。
        use_pil (bool): 是否使用 PIL 保存图像。如果为 False，则使用 torchvision 的 save_image（默认：False）。
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    for list_idx, tensor in enumerate(tensor_list):
        # 确保输入张量的形状为 (N, H, W, 3)
        if tensor.ndimension() != 4 or tensor.size(-1) != 3:
            raise ValueError(f"张量列表中的第 {list_idx} 个张量形状不为 (N, H, W, 3)。实际形状为 {tensor.shape}")

        N, H, W, C = tensor.shape

        for n in range(N):
            img_tensor = tensor[n]  # 形状: (H, W, 3)

            if use_pil:
                # 使用 PIL 保存图像
                # 转换为 (H, W, C) 的 NumPy 数组
                img_np = img_tensor.cpu().numpy()

                # 如果张量值范围在 [0,1]，则转换为 [0,255]
                if img_np.max() <= 1.0:
                    img_np = (255 * np.clip(img_np, 0, 1)).astype(np.uint8)
                else:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                # 创建 PIL 图像对象
                pil_img = Image.fromarray(img_np)

                # 定义保存路径
                save_path = os.path.join(output_dir, f"{prefix}_{list_idx}_{n}.png")

                # 保存图像
                pil_img.save(save_path)
    print(f"所有图像已成功保存到文件夹：{output_dir}")