import argparse
import torch
import os
import torch.optim.adamw
from torch.utils.data import TensorDataset, Dataset, DataLoader
from model import Model
from encoding import load_picture, encode
from utils import train,test
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    pos, label, width, height  = load_picture(args)
    encoded_pos = encode(pos,args.L)

    dataset = TensorDataset(encoded_pos, label)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=512, num_workers=2)

    model = Model(input_dimension=2+4*args.L, layer_num=args.layer_num)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train(dataloader, model, optimizer, scheduler, args, width, height)
    #test(width, height, model, args)


if __name__=="__main__":
    #免费提供-h选项
    parser = argparse.ArgumentParser(description='The configs')# description = Text to display before the argument help
    #均为可选参数，没有positional args;这些变量的名字就是去掉--后的字符串
    #positional args依顺序而定；-h的时候会打印出来哪个位置是什么变量
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--layer_num', type=int, default=10)
    parser.add_argument('--picture_dir', '-pd', type=str, required=True)
    parser.add_argument('--L', type=int, default=20, help='dimension of positional encoding')
    parser.add_argument('--checkpoint_save','-cs', type=str, default='2d_pos_encoding/checkpoint', help='Path to save checkpoint file')
    parser.add_argument('--checkpoint_load','-cl', type=str, default=None, help='Path to load checkpoint file, if None train from scratch')
    parser.add_argument('-v', action='store_true', help='Verbose mode')
    parser.add_argument('--output_dir','-od', type=str, default='2d_pos_encoding/result', help='Path to save output picture')
    parser.add_argument('--regularization','-reg',type=float, default=0, help='Regularization strength')
    args = parser.parse_args()

    picture_filename = os.path.splitext(os.path.basename(args.picture_dir))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_save, exist_ok=True)

    main(args)