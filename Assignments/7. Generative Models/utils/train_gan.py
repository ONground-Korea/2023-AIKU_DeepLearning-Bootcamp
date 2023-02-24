import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_generator(generator, discriminator, generated_img, optimizer, device):
    generator.zero_grad()
    loss = # TODO 
    loss.backward()
    optimizer.step()

    return loss.item()

def train_discriminator(discriminator, real_img, generated_img, optimizer, device):
    discriminator.zero_grad()
    loss = # TODO
    optimizer.step()

    # Clip weights of discriminator
    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)

    return loss.item()