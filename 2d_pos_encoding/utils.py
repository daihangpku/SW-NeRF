import tqdm #进度条
import os
import torch
import time
def load_checkpoint():#buggy
    checkpoint = torch.load(args.checkpoint_load)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
def save_checkpoint(model, optimizer, cur_epoch, metrics, filename="checkpoint"):#buggy
    os.makedirs(os.path.join(os.getcwd(),"checkpoint"), exist_ok=True)
    filename = os.path.join(os.getcwd(),"checkpoint",filename+".pth")
    checkpoint = {'cur_epoch': cur_epoch + 1,'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,'accuracy': accuracy}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch+1} to {filename}")

def train(dataloader, model, optimizer, args):
    epoch = 50
    cur_epoch = 0
    metrics = {"MSE":[],"PSNR":[]}
    if(args.checkpoint_load):
        load_checkpoint(...)
    epoch_iterator = range(cur_epoch,epoch) if args.v else tqdm(range(cur_epoch,epoch), desc="Epoch", ncols=100)
    dataloader = dataloader if args.v else tqdm(dataloader, desc="Batches", ncols=100,leave=False)
    start_time = time.time()
    for i in epoch_iterator:
        mse = 0
        gray_mse = 0
        gray_max = 0
        iternum = len(dataloader)
        for pos,color in dataloader:
            optimizer.zero_grad()
            output = model(pos)
            loss = torch.nn.functional.mse_loss(output, color)
            loss.backward()
            optimizer.step()
            # get mse
            mse += loss.item()
            # get psnr
            gray_color = 0.2989*color[:,0] + 0.5870*color[:,1] + 0.1140*color[:,2]
            gray_output = 0.2989*output[:,0] + 0.5870*output[:,1] + 0.1140*output[:,2]
            gray_mse += torch.nn.functional.mse_loss(gray_output, gray_color).item()
            gray_max = max(gray_max, torch.max(gray_color).item())
        mse /= iternum; gray_mse /= iternum
        psnr = 10*torch.log(gray_max**2/gray_mse)/torch.log(torch.tensor(10.0))
        metrics["MSE"].append(mse)
        metrics["PSNR"].append(psnr)
        if args.v:
            print(f"Epoch {i+1}/{epoch} MSE: {mse:.4f} PSNR: {psnr:.4f} time: {time.time()-start_time:.2f}s")
        save_checkpoint(model,optimizer,i,metrics)

def test():
    '''
    display a final result
    '''
    pass