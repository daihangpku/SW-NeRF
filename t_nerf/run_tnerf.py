# run.py
import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedder import *
from model import *  # 确保从 model.py 导入 TNeRF
from ray import *
from dataloader.load_blender_dnerf import load_blender_data
from utils import config_parser_dnerf, hsv_to_rgb
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

    def ret(inputs_pos, viewdirs, dyn_t):  # 修改: 接受 viewdirs 和 dyn_t 作为额外参数
        num_batches = inputs_pos.shape[0]  # 获取输入的批次大小
        out_list = []  # 存储输出结果的列表
        for i in range(0, num_batches, chunk):  # 按照指定的chunk大小进行分批次处理
            # 这里inputs_pos 的维度是pos_dim+view_dim，要在forward里面处理一下
            out= fn(inputs_pos[i:i+chunk], viewdirs[i:i+chunk], dyn_t[i:i+chunk])  # 修改: 传递 viewdirs 和 dyn_t
            out_list += [out]  # 将当前批次的输出添加到列表中
        return torch.cat(out_list, 0) # 合并所有批次的输出和梯度并返回

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
        embedded_times = embedded_time  # 修改: 仅使用一次时间嵌入
    else:
        raise NotImplementedError  # 如果不启用离散时间嵌入，抛出错误

    # 嵌入视角（方向）
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)  # 扩展视角的维度
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # 展平视角
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # 对视角进行嵌入
        embedded = torch.cat([embedded, embedded_dirs], -1)  # 将位置和视角嵌入拼接起来
    else:
        embedded_dirs = None

    # 应用网络并返回输出和位置梯度
    if embedded_dirs is not None:
        outputs_flat= batchify(fn, netchunk)(embedded, embedded_dirs, embedded_times)  # 修改: 传递 viewdirs 和 dyn_t
    else:
        outputs_flat = batchify(fn, netchunk)(embedded, embedded_dirs, embedded_times)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])  # 恢复输出的形状

    return outputs  # 返回输出和位置梯度


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
    near = near * torch.ones_like(rays_d[...,:1]).to(device)
    far = far * torch.ones_like(rays_d[...,:1]).to(device)
    frame_time = frame_time * torch.ones_like(rays_d[...,:1]).to(device) # 设置时间戳
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
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, rays=None, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)  # 修改: 将 rays 设置为 None 并传递 c2w
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

    # 修改: 设置输入特征维度为位置、视角和时间的嵌入维度之和
    net_dim = 128  # 或根据需要从 args 中获取
    depth = args.netdepth  # 根据需要从 args 中获取
    skip_layer = 4  # 或根据需要从 args 中获取

    # 实例化单个 TNeRF 模型
    model = TNeRF(
        depth=depth,
        in_feat=input_ch,
        dir_feat=input_ch_views,
        time_feat=input_ch_time,
        net_dim=net_dim,
        skip_layer=skip_layer
    ).to(device)  # 修改: 使用 TNeRF 替代原有的 NeRF

    grad_vars = list(model.parameters())  # 获取所有可训练的参数

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
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')  # 修改: 仅初始化单个模型

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
        if args.do_half_precision:
            amp.load_state_dict(ckpt['amp'])

    ##########################

    # 定义训练时的渲染参数
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : 0,  # 修改: 不使用额外的细化采样
        'network_fn' : model,  # 修改: 仅使用单个网络
        'N_samples' : args.N_samples,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer  # 修改: 不再返回 model_fine


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model. 3 for RGB and 1 for sigma.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)  # 修改: 将张量移动到 GPU

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).to(device) * raw_noise_std  # 修改: 将噪声移动到 GPU

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)  # 修改: 将噪声移动到 GPU

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,  # 修改: 不再使用 network_fine
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None, # 线性/逆深度采样
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
    rgb_map_0, disp_map_0, acc_map_0 = None, None, None

    if z_vals is None: #default
        #print("z_vals is None, using default sampling strategy")
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples]).to(device)

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape)) * raw_noise_std
                t_rand = torch.Tensor(t_rand).to(device)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


        if N_importance > 0:
            # 修改: 不使用两个模型，忽略 N_importance
            print("Warning: N_importance is set but only a single model is used.")
        #print(f"pts shape: {pts.shape}")
        raw = network_query_fn(pts, viewdirs, frame_time, network_fn)  # 修改: 只使用一个网络
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    else:
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        raw = network_query_fn(pts, viewdirs, frame_time, network_fn)  # 修改: 只使用一个网络
        rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals}
    if retraw:
        ret['raw'] = raw

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def train():

    parser = config_parser_dnerf()
    args = parser.parse_args()

    # Load data

    if args.dataset_type == 'blender':
        images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        # images = [rgb2hsv(img) for img in images]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 确保时间范围从0开始，到1结束
    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time == 0., "time must start at 0"
    assert max_time == 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_times = np.array(times[i_test])

    # Create log dir and copy the config file
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

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor, save_also_gt=True)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.N_iter + 1
    print('Begin')

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # 修改: 实现基于 viewdirs 的随机批次采样
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 3, 3]
            batch = torch.transpose(batch, 0, 1)  # [3, B, 3]
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            if i >= args.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                skip_factor = i / float(args.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])

            target = images[img_i]
            pose = poses[img_i, :3, :4]
            frame_time = times[img_i]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)  # 修改: 确保 pose 在 GPU 上

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH).to(device),  # 修改: 将 coords 转移到 GPU
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW).to(device)   # 修改: 将 coords 转移到 GPU
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H).to(device), torch.linspace(0, W-1, W).to(device)), -1)  # 修改: 将 coords 转移到 GPU

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)

        loss = img_loss
        psnr = mse2psnr(img_loss)

        # 修改: 移除对 rgb0 的处理，因为现在只有一个网络
        # if 'rgb0' in extras:
        #     img_loss0 = img2mse(extras['rgb0'], target_s)
        #     loss = loss + img_loss0
        #     psnr0 = mse2psnr(img_loss0)

        if args.do_half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if args.do_half_precision:
                save_dict['amp'] = amp.state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {img_loss.item()} PSNR: {psnr.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
            tqdm.write(tqdm_txt)

            writer.add_scalar('loss', img_loss.item(), i)
            writer.add_scalar('psnr', psnr.item(), i)
            if args.add_tv_loss:
                writer.add_scalar('tv', tv_loss.item(), i)

        del loss, img_loss, psnr, target_s
        if args.add_tv_loss:
            del tv_loss
        del rgb, disp, acc, extras

        if i%args.i_img==0:
            torch.cuda.empty_cache()
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            frame_time = times[img_i]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=None, c2w=pose, frame_time=frame_time,
                                                    **render_kwargs_test)  # 修改: 将 rays 设置为 None 并传递 c2w

            psnr = mse2psnr(img2mse(rgb, target))
            writer.add_image('gt', to8b(target.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('rgb', to8b(rgb.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('disp', disp.cpu().numpy(), i, dataformats='HW')
            writer.add_image('acc', acc.cpu().numpy(), i, dataformats='HW')

            # 修改: 移除对 rgb0 和 disp0 的处理，因为现在只有一个网络
            # if 'rgb0' in extras:
            #     writer.add_image('rgb_rough', to8b(extras['rgb0'].cpu().numpy()), i, dataformats='HWC')
            # if 'disp0' in extras:
            #     writer.add_image('disp_rough', extras['disp0'].cpu().numpy(), i, dataformats='HW')
            # if 'z_std' in extras:
            #     writer.add_image('acc_rough', extras['z_std'].cpu().numpy(), i, dataformats='HW')

            print("finish summary")
            writer.flush()

        if i%args.i_video==0:
            # Turn on testing mode
            print("Rendering video...")
            with torch.no_grad():
                savedir = os.path.join(basedir, expname, 'frames_{}_spiral_{:06d}_time/'.format(expname, i))
                rgbs, disps = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, savedir=savedir)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            print('Testing poses shape...', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), torch.Tensor(render_times[i_test]).to(device),
                            hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
