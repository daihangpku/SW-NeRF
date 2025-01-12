import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import vallina_NeRF as NeRF

from embedder import get_embedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    config = {}
    with open(config_path, 'r') as file:
        for line in file:
            try: key, value = line.strip().split(' = ')
            
            except: continue    
            print(key, value)
            config[key] = value
    return config

def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, input_dims=3, i=args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, input_dims=3, i=args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    return model, model_fine, optimizer, network_query_fn

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    if len(inputs.shape) == 2:
        inputs = inputs[:,None]
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        if inputs.shape != viewdirs.shape:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
        else:
            input_dirs  = viewdirs[:,None]

        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def load_checkpoint(ckpt_path, model, model_fine, optimizer):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['network_fn_state_dict'])
    if model_fine is not None:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start = ckpt['global_step']
    return start
import torch
def query_nerf(x, y, z, viewdir, model, embed_fn, embeddirs_fn, device):
    """
    Query the NeRF model with given inputs.
    
    Args:
        x, y, z: Spatial coordinates.
        viewdir: View direction vector.
        model: NeRF model.
        embed_fn: Embedding function for spatial coordinates.
        embeddirs_fn: Embedding function for view directions.
        device: Device to run the model on.
    
    Returns:
        r, g, b: Color values.
        rho: Density value.
    """
    # Prepare inputs
    inputs = torch.tensor([[x, y, z]], dtype=torch.float32).to(device)
    viewdirs = torch.tensor([viewdir], dtype=torch.float32).to(device)
    
    # Embed inputs
    embedded = embed_fn(inputs)
    embedded_dirs = embeddirs_fn(viewdirs)
    embedded = torch.cat([embedded, embedded_dirs], -1)
    
    # Query the model
    with torch.no_grad():
        outputs = model(embedded)
    
    # Extract color and density
    rgb = torch.sigmoid(outputs[..., :3]).cpu().numpy()
    rho = outputs[..., 3].cpu().numpy()
    
    return rgb[0, 0], rgb[0, 1], rgb[0, 2], rho[0]
from utils import config_parser
def load_model():
    config_path = 'configs/lego.txt'
    ckpt_path = 'logs/blender_paper_lego/150000.tar'
    parser = config_parser()
    args = parser.parse_args()
    #args = load_config(config_path)
    model, model_fine, optimizer, network_query_fn = create_nerf(args)
    start = load_checkpoint(ckpt_path, model, model_fine, optimizer)

    print(f"Model loaded from checkpoint {ckpt_path} at step {start}")
    return model, model_fine, optimizer, network_query_fn, args