import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models.VAE import VAE
from utils.train_vae import train_vae, test_vae
import os
from tqdm.auto import tqdm


def inference(epoch, generator, fixed_z, model, device):
    directory = f'results_{model}'
    generated_img = generator(fixed_z).detach()
    if model == 'vae':
        generated_img = generated_img.view(64, 1, 28, 28)
    if not os.path.exists(directory):
         os.makedirs(directory)
    save_image(generated_img, f'results_{model}/sample_' + str(epoch) + '.png')

    