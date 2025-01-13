import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import sys
import pyramid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedder import *
from model import *
from ray import *
from dataloader.load_blender_dnerf import load_blender_data
from utils import *
from pyramid import *
try:
    from apex import amp  # 尝试导入apex库（用于混合精度训练）
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU
np.random.seed(0)  # 设置NumPy的随机种子
DEBUG = False  # 是否启用调试模式


def batchify(fn, chunk):
    """构造一个适用于小批次的函数版本。
    fn：原始函数
    chunk：每次处理的数据块大小
    """
    if chunk is None:
        return fn  # 如果没有指定chunk，直接返回原始函数
    
    def ret(inputs_pos, inputs_time):
        num_batches = inputs_pos.shape[0]  # 获取输入的批次大小

        out_list = []  # 存储输出结果的列表
        dx_list = []  # 存储梯度的列表
        for i in range(0, num_batches, chunk):  # 按照指定的chunk大小进行分批次处理
            out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            out_list += [out]  # 将当前批次的输出添加到列表中
            dx_list += [dx]  # 将当前批次的梯度添加到列表中
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)  # 合并所有批次的输出和梯度并返回
    
    return ret  # 返回构造好的批次处理版本的函数


def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024*64,
                embd_time_discr=True):
    """准备输入并应用网络 'fn'。
    inputs: N_rays x N_points_per_ray x 3
    viewdirs: N_rays x 3
    frame_time: N_rays x 1
    """
    assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"  # 确保所有点的时间是相同的
    cur_time = torch.unique(frame_time)[0]  # 获取当前的时间

    # 嵌入位置（坐标）
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # 展平输入的空间坐标
    embedded = embed_fn(inputs_flat)  # 对位置进行嵌入

    # 嵌入时间
    if embd_time_discr:
        B, N, _ = inputs.shape  # 获取输入的批次大小和点数
        input_frame_time = frame_time[:, None].expand([B, N, 1])  # 扩展时间的维度
        input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])  # 展平时间
        embedded_time = embedtime_fn(input_frame_time_flat)  # 对时间进行嵌入
        embedded_times = [embedded_time, embedded_time]  # 时间嵌入的重复两次，用于后续计算

    else:
        assert NotImplementedError  # 如果不启用离散时间嵌入，抛出错误

    # 嵌入视角（方向）
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)  # 扩展视角的维度
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # 展平视角
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # 对视角进行嵌入
        embedded = torch.cat([embedded, embedded_dirs], -1)  # 将位置和视角嵌入拼接起来

    # 应用网络并返回输出和位置梯度
    outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, embedded_times)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])  # 恢复输出的形状
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])  # 恢复位置梯度的形状
    
    return outputs, position_delta  # 返回输出和位置梯度


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk): # 将光线分成更小的批次进行渲染
        ret = render_rays(rays_flat[i:i+chunk], **kwargs) # 渲染当前批次的光线
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # 合并所有批次的结果
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} # 将所有批次的结果拼接成一个大张量
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        # 如果提供了 c2w 转换矩阵，使用它来渲染完整图像
        rays_o, rays_d = get_rays(H, W, focal, c2w) # 获取光线的原点和方向
    else:
        # use provided ray batch
        # 如果没有提供 c2w，则使用给定的光线批次
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() # 展平光线原点
    rays_d = torch.reshape(rays_d, [-1,3]).float() # 展平光线方向

    # 为每条光线设置最近和最远距离
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    frame_time = frame_time * torch.ones_like(rays_d[...,:1]) # 设置时间戳
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1) # 将所有光线的原点、方向、最近距离、最远距离、时间戳合并为一个张量
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # 如果使用视角方向，将视角方向也拼接到光线数据中

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs) # 分批渲染光线
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # 提取需要的结果
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0):
    """
    渲染路径：用于渲染多个视角的图像，支持保存图像结果。
    
    参数:
      render_poses: 渲染时使用的相机姿态集合。
      render_times: 每一帧的时间戳。
      hwf: 图像的高宽焦距。
      chunk: 一次渲染的最大光线数，用于控制内存使用。
      render_kwargs: 渲染时的额外参数。
      gt_imgs: 如果有的话，真实的图像。
      savedir: 保存渲染结果的目录。
      render_factor: 渲染缩放因子（默认为0，即不缩放）。
      save_also_gt: 是否同时保存真实图像（默认为False）。
      i_offset: 索引偏移量，主要用于批次渲染时避免文件名冲突。

    返回:
      rgbs: 渲染得到的RGB图像数组。
      disps: 渲染得到的视差图（深度的逆）。
    """

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)

    rgbs = []
    disps = []

    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        # 渲染当前帧
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            # 如果指定了保存目录，保存渲染结果
            rgb8_estim = to8b(rgbs[-1])
            filename = os.path.join(save_dir_estim, '{:03d}.png'.format(i+i_offset))
            imageio.imwrite(filename, rgb8_estim)
            if save_also_gt:
                rgb8_gt = to8b(gt_imgs[i])
                filename = os.path.join(save_dir_gt, '{:03d}.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_gt)

    rgbs = np.stack(rgbs, 0) # 将所有渲染的RGB图像堆叠成一个数组
    disps = np.stack(disps, 0) # 将所有视差图堆叠成一个数组

    return rgbs, disps


def create_nerf(args):
    """
    实例化NeRF模型，设置模型的各项参数并返回。

    参数:
      args: 包含模型参数的配置对象。
      
    返回:
      render_kwargs_train: 训练时渲染的参数。
      render_kwargs_test: 测试时渲染的参数。
      start: 开始训练的步数。
      grad_vars: 模型的所有可训练参数。
      optimizer: 优化器对象。
    """
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed)
    embedtime_fn, input_ch_time = get_embedder(args.multires, 1, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4 # 如果需要重要性采样，则输出5通道
    skips = [4] # 跳跃层
    model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                 zero_canonical=not args.not_zero_canonical).to(device)
    grad_vars = list(model.parameters()) # 获取所有可训练的参数

    model_fine = None
    if args.use_two_models_for_fine:
        # 如果使用两个模型（粗模型和精细模型），则实例化精细模型
        model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                          zero_canonical=not args.not_zero_canonical).to(device)
        grad_vars += list(model_fine.parameters())

    # 定义网络查询函数
    network_query_fn = lambda inputs, viewdirs, ts, network_fn : run_network(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtime_fn=embedtime_fn,
                                                                netchunk=args.netchunk,
                                                                embd_time_discr=args.nerf_type!="temporal")

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    if args.do_half_precision:
        # 如果使用半精度计算，初始化AMP
        print("Run model at half precision")
        if model_fine is not None:
            [model, model_fine], optimizers = amp.initialize([model, model_fine], optimizer, opt_level='O1')
        else:
            model, optimizers = amp.initialize(model, optimizer, opt_level='O1')

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.do_half_precision:
            amp.load_state_dict(ckpt['amp'])

    ##########################

    # 定义训练时的渲染参数
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine': model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : args.use_two_models_for_fine,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


        if N_importance <= 0:
            raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals,
           'position_delta' : position_delta}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def reconstruct_and_compute_loss(pyramid_outputs, target_images):
    """
    从模型输出的拉普拉斯金字塔重建图像，并计算与目标图像的损失。
    :param pyramid_outputs: 模型输出的拉普拉斯金字塔列表，形状为 (layer_num, B, 3, H, W)
    :param target_images: 目标图像，形状为 (B, 3, H_orig, W_orig)
    :return: 重建图像、损失值和 PSNR
    """
    reconstructed = reconstruct_image_from_pyramid_batch(pyramid_outputs)
    target_images = target_images.permute(0, 2, 3, 1)  # 转为 (B, H, W, 3)
    loss = F.mse_loss(reconstructed, target_images)
    psnr = 10 * torch.log10(1 / loss)
    return reconstructed, loss, psnr

def train():
    parser = config_parser_dnerf()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    if args.dataset_type == 'blender':
        images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip
        )
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    images = torch.Tensor(images).to(device)
    print(f"images shape: {images.shape}")

    # 生成图像金字塔
    pyr_images = pyramid.generate_laplacian_pyramid_batch(images, levels=args.layer_num)
    # 假设形状为 (layer_num, N, 3, H, W)
    print(f"pyr_images shape: {[images.shape for images in pyr_images]}")

    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time == 0., "time must start at 0"
    assert max_time == 1., "max time must be 1"

    # 原始相机内参
    H_orig, W_orig, focal_orig = hwf
    H_orig, W_orig = int(H_orig), int(W_orig)

    # 为每一层计算缩放因子并调整相机内参
    pyr_hwf = []
    scale_factor = 2  # 例如，每层分辨率减半
    for layer in range(args.layer_num):
        scale = scale_factor ** layer
        H_l = H_orig // scale
        W_l = W_orig // scale
        focal_l = focal_orig / scale
        pyr_hwf.append([H_l, W_l, focal_l])
        print(f"Layer {layer}: H={H_l}, W={W_l}, focal={focal_l}")

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_times = np.array(times[i_test])

    # 创建日志目录并保存配置文件
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 创建多个 NeRF 模型，每个对应一个分辨率层
    render_kwargs_train_list = []
    render_kwargs_test_list = []
    grad_vars_list = []
    optimizer_list = []
    start_list = []
    global_steps = []

    for layer in range(args.layer_num):
        print(f"Creating NeRF model for layer {layer}")
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
        render_kwargs_train.update({'near': near, 'far': far})
        render_kwargs_test.update({'near': near, 'far': far})

        # 将模型移动到 GPU
        render_kwargs_train['network_fn'].to(device)
        if render_kwargs_train.get('network_fine') is not None:
            render_kwargs_train['network_fine'].to(device)
        render_kwargs_test['network_fn'].to(device)
        if render_kwargs_test.get('network_fine') is not None:
            render_kwargs_test['network_fine'].to(device)

        render_kwargs_train_list.append(render_kwargs_train)
        render_kwargs_test_list.append(render_kwargs_test)
        grad_vars_list.append(grad_vars)
        optimizer_list.append(optimizer)
        start_list.append(start)
        global_steps.append(start)

    # 移动测试数据到 GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    # 准备 raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    rays_rgb_list = []
    i_batch = []

    if use_batching:
        print('Generating rays for all layers')
        for layer in range(args.layer_num):
            H_l, W_l, focal_l = pyr_hwf[layer]
            rays = np.stack([get_rays_np(H_l, W_l, focal_l, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
            rays_rgb = np.concatenate([rays, pyr_images[layer][..., None].cpu().numpy()], 1)  # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N_train)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            np.random.shuffle(rays_rgb)
            rays_rgb = torch.Tensor(rays_rgb).to(device)
            rays_rgb_list.append(rays_rgb)
            i_batch.append(0)
        print('Rays generation completed')

    # 移动训练数据到 GPU
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)
    # pyr_images 已经在 GPU 上，无需再次移动

    N_iters = args.N_iter + 1
    print('Begin training')

    # 创建 SummaryWriter
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    # 定义图像重建函数
    # 已在前面定义

    for i in trange(start_list[0], N_iters):
        time0 = time.time()
        current_time = time0

        # 收集每个层级的模型输出
        pyramid_outputs = []

        # 针对每个层级进行多次训练
        for layer in range(args.layer_num):
            render_kwargs_train = render_kwargs_train_list[layer]
            optimizer = optimizer_list[layer]
            global_step = global_steps[layer]
            H_l, W_l, focal_l = pyr_hwf[layer]

            # 多次训练同一模型以增强局部性
            for _ in range(args.inner_iteration):  # inner_iteration 是一个新参数，定义每层每步的训练次数
                outputs = []
                if use_batching:
                    # 随机批次光线
                    B = N_rand
                    batch = rays_rgb_list[layer][i_batch[layer]:i_batch[layer]+N_rand]  # [B, 3, 3]
                    if batch.shape[0] < N_rand:
                        # 重新打乱并重置批次
                        rays_rgb_list[layer] = rays_rgb_list[layer][torch.randperm(rays_rgb_list[layer].shape[0])]
                        i_batch[layer] = 0
                        batch = rays_rgb_list[layer][i_batch[layer]:i_batch[layer]+N_rand]
                    batch_rays = batch[:, :2]  # [B, 2, 3] -> [B, rays_o, rays_d]
                    target_s = batch[:, 2]    # [B, 3]

                    i_batch[layer] += N_rand

                else:
                    # 从单个图像中随机采样光线
                    img_i = np.random.choice(i_train)
                    target = pyr_images[layer][img_i]
                    pose = poses[img_i, :3, :4]
                    frame_time = times[img_i]

                    rays_o, rays_d = get_rays(H_l, W_l, focal_l, pose)  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H_l // 2 * args.precrop_frac)
                        dW = int(W_l // 2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H_l // 2 - dH, H_l // 2 + dH - 1, 2 * dH),
                                torch.linspace(W_l // 2 - dW, W_l // 2 + dW - 1, 2 * dW)
                            ), -1)
                        if i == start_list[layer]:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(0, H_l - 1, H_l),
                                torch.linspace(0, W_l - 1, W_l)
                            ), -1
                        )  # (H, W, 2)

                    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o_batch, rays_d_batch], 0)  # [2, N_rand, 3]
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                ##### 核心优化循环 #####
                rgb, disp, acc, extras = render(
                    H_l, W_l, focal_l, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                    verbose=(i < 10), retraw=True,
                    **render_kwargs_train
                )
                # 收集输出到拉普拉斯金字塔
                outputs.append(rgb)  # [B, 3, H, W]

                optimizer.zero_grad()
                img_loss = F.mse_loss(rgb, target_s)

                loss = img_loss
                psnr = 10 * torch.log10(1 / img_loss)

                if 'rgb0' in extras:
                    img_loss0 = F.mse_loss(extras['rgb0'], target_s)
                    loss = loss + img_loss0
                    psnr0 = 10 * torch.log10(1 / img_loss0)
                loss *= (1-args.loss_decrease_rate)**i
                if args.do_half_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                # 更新学习率
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                # 日志记录
                if _ % args.i_print == 0:
                    tqdm_txt = f"[TRAIN] Layer: {layer} Iter: {i} Loss_fine: {img_loss.item()} PSNR: {psnr.item()}"
                    if 'rgb0' in extras:
                        tqdm_txt += f" Loss0: {img_loss0.item()} PSNR0: {psnr0.item()}"
                    tqdm.write(tqdm_txt)

                    writer.add_scalar(f'loss_layer_{layer}', img_loss.item(), i)
                    writer.add_scalar(f'psnr_layer_{layer}', psnr.item(), i)
                    if 'rgb0' in extras:
                        writer.add_scalar(f'loss0_layer_{layer}', img_loss0.item(), i)
                        writer.add_scalar(f'psnr0_layer_{layer}', psnr0.item(), i)

                del loss, img_loss
                if 'rgb0' in extras:
                    del img_loss0
                del rgb, disp, acc, extras

                # 清理 GPU 缓存
                if _ % args.i_img == 0:
                    torch.cuda.empty_cache()
                outputs = torch.stack(outputs,dim=0)
                pyramid_outputs.append(outputs)
            global_steps[layer] += 1

        ##### 重建图像并计算整体损失 #####
        
        if i >= args.global_optimization_epoch:
            # 进行全局优化，包括重建图像的损失
            reconstructed_images, global_loss, global_psnr = reconstruct_and_compute_loss(pyramid_outputs, images[i_train])

            # 将全局损失反向传播
            # 假设所有模型的参数都在各自的 optimizer 中，需确保梯度正确传递
            global_loss.backward()

            # 更新所有优化器
            for optimizer in optimizer_list:
                optimizer.step()

            # 日志记录全局损失和 PSNR
            if i % args.i_print == 0:
                tqdm_txt = f"[GLOBAL OPT] Iter: {i} Global Loss: {global_loss.item()} Global PSNR: {global_psnr.item()}"
                tqdm.write(tqdm_txt)

                writer.add_scalar('global_loss', global_loss.item(), i)
                writer.add_scalar('global_psnr', global_psnr.item(), i)

        ##### 保存所有模型 #####
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, f'{i:06d}.tar')
            save_dict = {
                'global_step': i,
            }
            for layer in range(args.layer_num):
                save_dict[f'network_fn_{layer}'] = render_kwargs_train_list[layer]['network_fn'].state_dict()
                if render_kwargs_train_list[layer].get('network_fine') is not None:
                    save_dict[f'network_fine_{layer}'] = render_kwargs_train_list[layer]['network_fine'].state_dict()
                save_dict[f'optimizer_{layer}'] = optimizer_list[layer].state_dict()

            if args.do_half_precision:
                save_dict['amp'] = amp.state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        ##### 生成和保存视频 #####
        if i % args.i_video == 0:
            print("Rendering video...")
            with torch.no_grad():
                # 收集所有模型的渲染输出
                pyramid_rendered = []
                for layer in range(args.layer_num):
                    rgbs, _ = render_path(
                        render_poses, render_times, hwf, args.chunk, render_kwargs_test_list[layer],
                        savedir=os.path.join(basedir, expname, f'frames_layer_{layer}_{i:06d}_time/')
                    )
                    pyramid_rendered.append(torch.Tensor(rgbs).to(device))  # 假设 rgbs 是 (num_frames, H, W, 3)
                # 重建图像
                pyramid_rendered = torch.stack(pyramid_rendered, dim=0)  # [layer_num, num_frames, 3, H, W]
                reconstructed_video = reconstruct_image_from_pyramid_batch(pyramid_rendered)  # [num_frames, H, W, 3]

                # 保存视频
                moviebase = os.path.join(basedir, expname, f'{expname}_reconstructed_{i:06d}_')
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(reconstructed_video.cpu().numpy()), fps=30, quality=8)
            print('Done, saving reconstructed video')

        ##### 测试集渲染 #####
        if i % args.i_testset == 0:
            testsavedir = os.path.join(basedir, expname, f'testset_{i:06d}')
            print('Testing poses shape...', poses[i_test].shape)
            with torch.no_grad():
                # 收集所有模型的渲染输出
                pyramid_test = []
                for layer in range(args.layer_num):
                    rgbs, _ = render_path(
                        torch.Tensor(poses[i_test]).to(device), torch.Tensor(render_times).to(device),
                        hwf, args.chunk, render_kwargs_test_list[layer],
                        gt_imgs=pyr_images[layer][i_test], savedir=os.path.join(testsavedir, f'layer_{layer}')
                    )
                    pyramid_test.append(torch.Tensor(rgbs).to(device))
                # 重建图像
                pyramid_test = torch.stack(pyramid_test, dim=0)  # [layer_num, test_N, 3, H, W]
                reconstructed_test = reconstruct_image_from_pyramid_batch(pyramid_test)  # [test_N, H, W, 3]

                # 保存测试集重建图像
                # 这里可以根据需求进一步处理，比如保存为图像文件或其他格式
            print('Saved test set reconstructed images')

        ##### 输出各分辨率图像及融合结果 #####
        if i % args.i_img == 0:
            with torch.no_grad():
                # 收集各层级的图像输出
                individual_images = []
                for layer in range(args.layer_num):
                    # 选择一个验证集图像
                    img_i = np.random.choice(i_val)
                    target = pyr_images[layer][img_i]
                    pose = poses[img_i, :3, :4]
                    frame_time = times[img_i]

                    # 渲染
                    rgb, disp, acc, extras = render(
                        pyr_hwf[layer][0], pyr_hwf[layer][1], pyr_hwf[layer][2],
                        chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                        **render_kwargs_test_list[layer]
                    )

                    # 添加到列表
                    individual_images.append(to8b(rgb.cpu().numpy()))

                # 重建图像
                pyramid_rendered = torch.stack([torch.tensor(img).permute(2, 0, 1) for img in individual_images], dim=0).unsqueeze(0)  # [1, layer_num, 3, H, W]
                reconstructed_image = reconstruct_image_from_pyramid_batch(pyramid_rendered).squeeze(0).cpu().numpy()  # [H, W, 3]

                # 转换为可视化格式
                individual_images_np = [img for img in individual_images]
                fused_image_np = to8b(reconstructed_image)

                # 将各层图像和融合图像一起记录到 TensorBoard
                for layer in range(args.layer_num):
                    writer.add_image(f'individual_layer_{layer}', individual_images_np[layer], i, dataformats='HWC')
                writer.add_image('fused_image', fused_image_np, i, dataformats='HWC')

                print(f"Iteration {i}: Saved individual and fused images to TensorBoard")

        ##### 全局步数更新和日志 #####
        dt = time.time() - time0
        # print(f"Iteration {i} done, time: {dt}")

    # 训练结束后关闭 SummaryWriter
    writer.close()
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
