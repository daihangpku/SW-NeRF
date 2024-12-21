import numpy as np
import torch
from PIL import Image
def load_picture(args):
    '''
    input: args(picture directory)
    output: 
    a tensor of position, shape = (H*W),2
    and a tensor of color, shape = (H*W),3
    '''
    image = Image.open(args.picture_dir)
    image = image.convert('RGB')
    width, height = image.size
    pos = np.array([[x, y] for y in range(height) for x in range(width)])
    positions = torch.tensor(pos, dtype=torch.float32)

    color = np.array(list(image.getdata()))
    colors = torch.tensor(color, dtype=torch.float32)
    return positions, colors

def encode(pos,args):        #params:L
    '''
    encode [x,y] -> [x,y,...]
    output: a tensor of encoded position, shape = (H*W),4*args.L+2
    '''
    L = args.L 
    encoding = torch.zeros((pos.shape[0], 4 * L + 2)) 
    for i in range(L):
        # x 坐标的sin和cos
        encoding[:, 2 * i] = torch.sin(2**i * np.pi * pos[:, 0])
        encoding[:, 2 * i + 1] = torch.cos(2**i * np.pi * pos[:, 0])
        
        # y 坐标的sin和cos
        encoding[:, 2 * L + 2 * i] = torch.sin(2**i * np.pi * pos[:, 1])
        encoding[:, 2 * L + 2 * i + 1] = torch.cos(2**i * np.pi * pos[:, 1])
        
    return encoding