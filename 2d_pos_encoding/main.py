import argparse
import torch
import os
from torch.utils.data import TensorDataset, Dataset, DataLoader
from model import Model
from encoding import load_picture, encode
from utils import train,test
def main(args):
    pos, label = load_picture(args)
    encoded_pos = encode(pos,args)

    dataset = TensorDataset(encoded_pos, label)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=2)

    model = Model(input_dimension=2+4*args.L, layer_num=args.layer_num)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

    train(dataloader, model, optimizer, args)
    test(pos,model,args)


if __name__=="__main__":
    #免费提供-h选项
    parser = argparse.ArgumentParser(description='The configs')# description = Text to display before the argument help
    #均为可选参数，没有positional args;这些变量的名字就是去掉--后的字符串
    #positional args依顺序而定；-h的时候会打印出来哪个位置是什么变量
    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--layer_num', type=int, default=5)
    parser.add_argument('--picture_dir', type=str, required=True)
    parser.add_argument('--L', type=int, default='dimension of positional encoding')
    parser.add_argument('--checkpoint_save','-cs', type=str, default='checkpoint/', help='Path to save checkpoint file')
    parser.add_argument('--checkpoint_load','-cl', type=str, default=None, help='Path to load checkpoint file, if None train from scratch')
    parser.add_argument('-v', action='store_true', help='Verbose mode')
    args = parser.parse_args()
    
    main(args)