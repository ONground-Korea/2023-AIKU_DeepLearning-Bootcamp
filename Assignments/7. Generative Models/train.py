import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation
from IPython.display import HTML
from models.VAE import VAE
from models.GAN import weights_init, Generator, Discriminator
from utils.train_vae import train_vae, test_vae
from utils.train_gan import train_discriminator, train_generator
from utils.inference import inference
from losses import vae_loss_function
import pdb

########################################################################################################################
# python train.py --model vae --mode train
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=['vae', 'gan'], type=str, help="Choose the model you want to train")
parser.add_argument("--mode", choices=['train', 'eval'], default='train', type=str, help="mode for train or eval")
args = parser.parse_args()
########################################################################################################################

bs = 512  # batch size
epochs = 64 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST Dataset
train_dataset = datasets.MNIST(root='mnist/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='mnist/', train=False, transform=transforms.ToTensor(), download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# Model
if args.model == 'vae':
    vae = VAE(x_dim=784, h_dim=400, z_dim=20).to(device)
    print(vae)
elif args.model == 'gan':
    nc = 1  # 입력 이미지의 색 채널개수
    nz = 100  # latent z의 원소 개수
    ngf = 64  # generator를 통과할때 만들어질 특징 데이터의 채널개수
    ndf = 64  # discriminator를 통과할때 만들어질 특징 데이터의 채널개수
    
    generator = Generator(nc, nz, ngf).to(device)
    discriminator = Discriminator(nc, ndf).to(device)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print(generator)
    print(discriminator)


# optimizer
if args.model == 'vae':
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
elif args.model == 'gan':
    optimizer_generator = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)


# train
if args.mode == 'train':
    if args.model == 'vae':
        fixed_z = torch.randn(64, 20, device=device)  # 일정한 z를 설정함으로써 모델이 어떻게 학습하는지 보기 위함
        progress = tqdm(range(epochs))
        for epoch in progress:
            train_loss = train_vae(vae, optimizer, vae_loss_function, train_loader, device)
            test_loss = test_vae(vae, test_loader, vae_loss_function, device)
            progress.set_description(f'[Epoch {epoch}] train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}')
            inference(epoch, vae.decoder, fixed_z, args.model, device)

        torch.save(vae.state_dict(), "vae.pt")
            
    elif args.model == 'gan':
        progress = tqdm(range(epochs))
        fixed_z = torch.randn(64, nz, 1, 1, device=device)  # 일정한 z를 설정함으로써 모델이 어떻게 학습하는지 보기 위함
        for epoch in progress: 
            for i, data in enumerate(train_loader, 0):
                real_img = data[0].to(device)
                z = torch.randn(bs, nz, 1, 1, device=device)
                generated_img = generator(z)

                D_loss = train_discriminator(discriminator, real_img, generated_img, optimizer_discriminator, device)
                G_loss = train_generator(generator, discriminator, generated_img, optimizer_generator, device)
                
            progress.set_description(f'[Epoch {epoch}] D_loss: {D_loss:.4f}, G_loss: {G_loss:.4f}')
            inference(epoch, generator, fixed_z, args.model, device)
        
        torch.save(generator.state_dict(), "generator.pt")
        torch.save(discriminator.state_dict(), "discriminator.pt")

