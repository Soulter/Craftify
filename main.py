#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image
import argparse

from model_dcgan import GeneratorDCGAN, DiscriminatorDCGAN

def load_dataset(data_path_root: str, image_size: int = 64, batch_size: int = 128, workers: int = 2) -> torch.utils.data.DataLoader:
    '''
    Load dataset from path
    '''
    dataset = dset.ImageFolder(root=data_path_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    return dataloader

def load_device(ngpu: int = 1) -> torch.device:
    '''
    Load device
    '''
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("we will use", device)
    return device

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def _preview_dataset(dataloader: torch.utils.data.DataLoader, 
                     device: torch.device,
                     train_images_path: str = "training_images.jpg"):
    '''
    preview our dataset 64 grid
    '''
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(train_images_path)
    
def train(ngpu, nc, nz, ndf, ngf, device, lr, beta1, dataloader, num_epochs, save_path_dir: str = "pth_release/", tag: str = "t1_"):
    netG = GeneratorDCGAN(ngpu, nz, ngf, nc).to(device)
    netD = DiscriminatorDCGAN(ngpu, nc, ndf).to(device)
    print(netG)
    print(netD)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    
    # BCELoss
    criterion = nn.BCELoss()
    
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
            
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Training Loop...")
    
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # all fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    if not save_path_dir.endswith("/"):
        save_path_dir += "/"
    save_path = f"{save_path_dir}{tag}"
    torch.save(netD.state_dict(), f"{save_path}d.pth")
    torch.save(netG.state_dict(), f"{save_path}g.pth")
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.jpg")
    
    return netG, netD, f"{save_path}d.pth", f"{save_path}g.pth"

def handle_predict(netG: GeneratorDCGAN, nz, device, with_pth = None, num = 1, num_in_a_pic: int = 64):
    if with_pth:
        print(f"loading {with_pth}...")
        netG.load_state_dict(torch.load(with_pth))
    print("predicting...")
    list = []
    for i in range(num):
        noise = torch.randn(num_in_a_pic, nz, 1, 1, device=device)
        with torch.no_grad():
            output = netG(noise).detach().cpu()
        list.append(output)
    return list

def handle_output_head(outputs: list[torch.Tensor], path_dir: str = "output/"):
    i = 0
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    for output in outputs:
        # resize to 8*8
        output = nn.functional.interpolate(output, size=(8, 8), mode='bilinear', align_corners=False)
        grid = vutils.make_grid(output, padding=2, normalize=True)
        vutils.save_image(grid, f"{path_dir}/output_head_{i}.png")
        i += 1

def handle_output_all(outputs: list[torch.Tensor], mask = None, path_dir: str = "output/"):
    '''
    将网络的输出进行后处理
    
    output shape: (n, 3, 64, 64)
    '''
    
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    # 创建n个4通道的图像
    print("outputs", len(outputs))
    for j in range(len(outputs)):
        output = outputs[j]
        n = output.shape[0]
        output_4 = torch.zeros(n, 4, 64, 64)
        
        # 将mask应用到每个图像上
        for i in range(n):
            output_4[i][0] = output[i][0] * mask if mask else output[i][0]
            output_4[i][1] = output[i][1] * mask if mask else output[i][1]
            output_4[i][2] = output[i][2] * mask if mask else output[i][2]
            # output_4[i][3] = torch.tensor(mask)

        # 输出
        for o in output_4:
            vutils.save_image(o, f"output/output_test_{j}.png", normalize=True)
            j+=1
        
def train_flow(**kwargs) -> tuple[GeneratorDCGAN, DiscriminatorDCGAN]:
    # Set random seed for reproducibility
    manualSeed = kwargs.get("manualSeed", 999)
    
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    
    dataroot = kwargs.get("dataroot", "data/betterskinshead/")
    workers = kwargs.get("workers", 2)
    batch_size = kwargs.get("batch_size", 128)
    image_size = kwargs.get("image_size", 64)
    ngpu = kwargs.get("ngpu", 1)
    nc = kwargs.get("nc", 3)
    nz = kwargs.get("nz", 100)
    ndf = kwargs.get("ndf", 64)
    ngf = kwargs.get("ngf", 64)
    num_epochs = kwargs.get("num_epochs", 6)
    lr = kwargs.get("lr", 0.001)
    beta1 = kwargs.get("beta1", 0.6)
    
    data_loader = load_dataset(dataroot, image_size, batch_size, workers)
    device = load_device(ngpu)
    _preview_dataset(data_loader, device)
    netG, netD, netG_path, netD_path = train(ngpu, 
                                    nc, 
                                    nz, 
                                    ndf, 
                                    ngf, 
                                    device, 
                                    lr, 
                                    beta1, 
                                    data_loader, 
                                    num_epochs,
                                    tag="t1_")
    return locals()

def predict_flow(netG: GeneratorDCGAN, nz, device, with_pth = None, num = 1, num_in_a_pic: int = 64, type='head'):
    outputs = handle_predict(netG, nz, device, with_pth, num, num_in_a_pic)
    if type == 'head':
        handle_output_head(outputs)
    elif type == 'all':
        handle_output_all(outputs)
    else:
        raise ValueError("argument type should be 'head' or 'all'")
 
if __name__ == "__main__":
    # handle args
    args = argparse.ArgumentParser()
    args.add_argument("--dataroot", type=str, default="data/skin/SkinsOnlyHead")
    args.add_argument("--type", type=str, default="head")
    args.add_argument("--onlypredict", type=bool, default=False)
    args.add_argument("--gmodelpath", type=str, default="pth_release/t1_g_head.pth")
    
    params = {
        "dataroot": None,
        "workers": 2,
        "batch_size": 128,
        "image_size": 64,
        "ngpu": 1,
        "nc": 3,
        "nz": 100,
        "ndf": 64,
        "ngf": 64,
        "num_epochs": 6,
        "lr": 0.001,
        "beta1": 0.6,
        "data_loader": None,
        "device": None,
        "netG": None,
        "netD": None,
        "netG_path": "pth_release/t1_g_head.pth",
    }
    
    params['dataroot'] = args.parse_args().dataroot
    if params['dataroot'] is None:
        raise ValueError("dataroot argument is required")
    
    train_type = args.parse_args().type
    onlypredict = args.parse_args().onlypredict
    params['netG_path'] = args.parse_args().gmodelpath
    
    
    if not onlypredict:
        params1 = train_flow(**params)
        params.update(params1)
    
    num = 1
    num_in_a_pic = 64
    if params.get("device"):
        device = params.get("device")
    else:
        device = load_device(params.get("ngpu"))
    print("device", device)
    
    if params.get("netG"):
        netG = params.get("netG")
    else:
        netG = GeneratorDCGAN(params.get("ngpu"), 
                              params.get("nz"), 
                              params.get("ngf"), 
                              params.get("nc")).to(device)

    if params.get("netG_path"):
        netG_path = params.get("netG_path")
    else:
        netG_path = None
    predict_flow(netG, params['nz'], device, netG_path, num, num_in_a_pic, type=train_type)
    
