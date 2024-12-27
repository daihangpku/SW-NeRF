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

def save_checkpoint(model, optimizer, cur_epoch, metrics, args):#buggy
    picture_filename = os.path.splitext(os.path.basename(args.picture_dir))[0]
    filename = os.path.join(args.checkpoint_save,f"{picture_filename}_{args.L}_{args.layer_num}.pth")
    checkpoint = {'cur_epoch': cur_epoch + 1,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'metrics': metrics}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {cur_epoch+1} to {filename}")

def train(dataloader, model, optimizer, scheduler, args, width, height):
    
    epoch = args.epochs
    cur_epoch = 0
    metrics = {"MSE":[],"PSNR":[]}
    if(args.checkpoint_load):
        load_checkpoint(model,optimizer,args)
        cur_epoch, metrics = load_checkpoint(model,optimizer,args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    for i in range(cur_epoch,epoch):
        model.train()
        total_loss = 0
        total_mse = 0
        total_gray_mse = 0
        #gray_max = 0
        iternum = len(dataloader)
        for pos, color in tqdm.tqdm(dataloader, desc=f"epoch {i+1}", ncols=100):
            pos = pos.to(device)
            color = color.to(device)

            optimizer.zero_grad()
            output = model(pos)
            loss = torch.nn.functional.mse_loss(output, color)
            loss.backward()
            optimizer.step()

            # 计算 MSE
            total_mse += loss.item()
            gray_color = 0.2989*color[:,0] + 0.5870*color[:,1] + 0.1140*color[:,2]
            gray_output = 0.2989*output[:,0] + 0.5870*output[:,1] + 0.1140*output[:,2]
            total_gray_mse += torch.nn.functional.mse_loss(gray_output, gray_color).item()
            #gray_max = max(gray_max, torch.max(gray_color).item())

        avg_mse = total_mse / iternum
        avg_gray_mse = total_gray_mse / iternum
        #gray_max = torch.tensor(gray_max, device=device)
        avg_gray_mse = torch.tensor(avg_gray_mse, device=device)
        psnr = 10 * torch.log(1.0 ** 2 / avg_gray_mse) / torch.log(torch.tensor(10.0))

        metrics["MSE"].append(avg_mse)
        metrics["PSNR"].append(psnr)
        if args.v:
            print(f"Epoch {i+1}/{epoch} MSE: {avg_mse:.4f} PSNR: {psnr:.4f} time: {time.time()-start_time:.2f}s")
        
        save_checkpoint(model, optimizer, i, metrics, args)
        if((i+1)%20 == 0):
            test(width, height, model, args)

        scheduler.step()
    picture_filename = os.path.splitext(os.path.basename(args.picture_dir))[0]
    get_graph(metrics,f"{picture_filename}_{args.L}_{args.layer_num}",args)
    print(f"final mse: {metrics['MSE'][-1]}, final psnr: {metrics['PSNR'][-1]}")

def test(width, height, model, args):
    # possible bug:need to change color channels
    '''
    display a final result
    '''
    model.eval()
    picture = get_picture(width, height, model, args)
    picture_filename = os.path.splitext(os.path.basename(args.picture_dir))[0]
    output_dir = os.path.join(args.output_dir, f"{picture_filename}_{args.L}_{args.layer_num}.png")

    plt.imsave(output_dir,picture)
    # if(args.v):
    #     plt.imshow(picture)
    #     plt.show()

def get_picture(width, height, model, args):
    H = height
    W = width
    device = next(model.parameters()).device
    picture = np.zeros((H, W, 3))
    
    # 创建所有位置的网格
    positions = np.array([[j, i] for i in range(H) for j in range(W)], dtype=np.float32)
    positions = torch.tensor(positions)
    
    # 批量编码位置
    encoded_positions = encode(positions, args.L).to(device)
    
    # 批量预测
    with torch.no_grad():
        outputs = model(encoded_positions).cpu().numpy()
    
    # 将输出重塑为图片
    picture = outputs.reshape(H, W, 3)

    # 将图片的值截断到0到1的范围
    picture = np.clip(picture, 0, 1)

    return picture

def get_graph(metrics, filename, args):
    os.makedirs(os.path.join('2d_pos_encoding',"metrics"), exist_ok=True)
    for metric, values in metrics.items():
        plt.figure()
        if(args.v):
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            elif isinstance(values, list):
                values = np.array([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values])
            plt.plot(values)
        # 每隔20个epoch标注一次
            # for i in range(0, len(values), 20):
            #     plt.plot(i, values[i], 'ro')  # 红点标记
            #     plt.annotate(f'{values[i]:.4f}', 
            #         xy=(i, values[i]), 
            #         xytext=(0, 10),
            #         textcoords='offset points',
            #         ha='center',
            #         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            # 标注最终值（如果最后一个点不是20的倍数）
            if True:
                plt.plot(len(values)-1, values[-1], 'go')
                plt.annotate(f'{values[-1]:.4f}', 
                    xy=(len(values)-1, values[-1]), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.savefig(os.path.join('2d_pos_encoding',"metrics",filename+f'_{metric}.png'))
        plt.close()