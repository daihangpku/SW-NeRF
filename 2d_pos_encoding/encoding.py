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
    L = args.L 
    encoding = torch.zeros((pos.shape[0], 4 * L + 2)) 
    for i in range(L):
        encoding[:, 2 * i] = torch.sin(pos[:, 0] * (2 ** i))  
        encoding[:, 2 * i + 1] = torch.cos(pos[:, 0] * (2 ** i))  
        encoding[:, 2 * L + 2 * i] = torch.sin(pos[:, 1] * (2 ** i))  
        encoding[:, 2 * L + 2 * i + 1] = torch.cos(pos[:, 1] * (2 ** i)) 
    return encoding