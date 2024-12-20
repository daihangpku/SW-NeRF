import argparse
import torch
import os
from torch.utils.data import TensorDataset, Dataset, DataLoader
from model import Model
from encoding import load_picture, encode
from utils import train
def main(args):
    pos, label = load_picture(args)
    encoded_pos = encode(pos)

    dataset = TensorDataset(encoded_pos, label)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=2)

    model = Model(input_dimension=2+4*args.L, layer_num=args.layer_num)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

    train(dataloader, model, optimizer, args)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--layer_num', type=int, default=5)
    parser.add_argument('--picture_dir', type=str, required=True)
    parser.add_argument('--L', type=str, default='dimension of positional encoding')
    parser.add_argument('--checkpoint_save','-cs', type=str, default='checkpoint/', help='Path to save checkpoint file')
    parser.add_argument('--checkpoint_load','-cl', type=str, default=None, help='Path to load checkpoint file, if None train from scratch')
    args = parser.parse_args()
    
    main(args)