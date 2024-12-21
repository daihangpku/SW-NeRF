import tqdm #进度条
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from encoding import encode
def load_checkpoint(model,optimizer,args):#buggy
    checkpoint = torch.load(args.checkpoint_load)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    cur_epoch = checkpoint['cur_epoch']
    metrics = checkpoint['metrics']
    return cur_epoch,metrics
def save_checkpoint(model, optimizer, cur_epoch, metrics, filename="checkpoint"):#buggy
    os.makedirs(os.path.join(os.getcwd(),"checkpoint"), exist_ok=True)#创建目录；如果目录已存在，则不会出现异常
    filename = os.path.join(os.getcwd(),"checkpoint",filename+".pth")
    checkpoint = {'cur_epoch': cur_epoch + 1,'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'metrics': metrics}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {cur_epoch+1} to {filename}")

def train(dataloader, model, optimizer, args):
    epoch = 50
    cur_epoch = 0
    metrics = {"MSE":[],"PSNR":[]}
    if(args.checkpoint_load):
        load_checkpoint(model,optimizer,args)
        cur_epoch,metrics = load_checkpoint(model,optimizer,args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    for i in range(cur_epoch,epoch):
        iternum = len(dataloader)
        for pos,color in tqdm.tqdm(dataloader, desc="Batches", ncols=100,leave=False):
            optimizer.zero_grad()
            output = model(pos)
            loss = torch.nn.functional.mse_loss(output, color)
            loss.backward()
            optimizer.step()
        mse = get_mse(dataloader,model)
        psnr = get_mse(dataloader,model)
        metrics["MSE"].append(mse)
        metrics["PSNR"].append(psnr)
        if args.v:
            print(f"Epoch {i+1}/{epoch} MSE: {mse:.4f} PSNR: {psnr:.4f} time: {time.time()-start_time:.2f}s")
        filename = f"cp_{i+1}_{args.L}_{args.layer_num}"
        save_checkpoint(model,optimizer,i,metrics,filename)
    get_graph(metrics,f"{args.L}_{args.layer_num}",args)
    print(f"final mse: {metrics['MSE'][-1]}, final psnr: {metrics['PSNR'][-1]}")
def test(pos:np.ndarray,model,args):
    # possible bug:need to change color channels
    '''
    display a final result
    '''
    picture = get_picture(pos,model,args)
    plt.imsave(f"{args.L}_{args.layer_num}.png",picture)
    if(args.v):
        plt.imshow(picture)

def get_mse(dataloader,model)->float:
    '''
    get mse of the model output picture
    '''
    mse = 0
    iternum = len(dataloader)
    for pos,color in dataloader:
        output = model(pos)
        mse += torch.nn.functional.mse_loss(output, color).item()
    mse /= iternum
    return mse
def get_psnr(dataloader,model)->float:
    '''
    get psnr of the model output picture
    '''
    gray_mse = 0
    gray_max = 0
    iternum = len(dataloader)
    for pos,color in dataloader:
        output = model(pos)
        gray_color = 0.2989*color[:,0] + 0.5870*color[:,1] + 0.1140*color[:,2]
        gray_output = 0.2989*output[:,0] + 0.5870*output[:,1] + 0.1140*output[:,2]
        gray_mse += torch.nn.functional.mse_loss(gray_output, gray_color).item()
        gray_max = max(gray_max, torch.max(gray_color).item())
    gray_mse /= iternum
    psnr = 10*torch.log(gray_max**2/gray_mse)/torch.log(torch.tensor(10.0))
    return psnr
def get_picture(pos,model,args):
    H = pos[:,0].max().item()+1
    W = pos[:,1].max().item()+1
    picture = np.ndarray((H,W,3))
    for i in range(H):
        for j in range(W):
            picture[i,j] = model(encode(torch.tensor([i,j]).view(1,-1),args)).detach().numpy().reshape(-1)
    return picture
def get_graph(metrics, filename,args):
    os.makedirs(os.path.join(os.getcwd(),"metrics"), exist_ok=True)
    for metric, values in metrics.items():
        plt.figure()
        if(args.v):
            plt.plot(values)
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.savefig(os.path.join(os.getcwd(),"metrics",filename+f'_{metric}.png'))
        plt.close()