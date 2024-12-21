import numpy as np
import torch
def load_picture(args):
    '''
    input: args(picture directory)
    output: 
    a tensor of position, shape = (H*W),2
    and a tensor of color, shape = (H*W),3
    '''
    H = 400;W = 300
    x = np.arange(H)
    y = np.arange(W)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    positions = np.stack([xv, yv], axis=-1).reshape(-1, 2)
    
    # Generate random colors for each position
    colors = torch.zeros(H * W, 3)
    
    return positions, colors

def encode(pos,args):        #params:L
    '''
    encode [x,y] -> [x,y,...]
    output: a tensor of encoded position, shape = (H*W),4*args.L+2
    '''
    return torch.zeros((pos.shape[0],4*args.L+2))
