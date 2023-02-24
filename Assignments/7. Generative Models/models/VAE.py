import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, z_dim)
        self.fc2_logvar = nn.Linear(h_dim, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def encoder(self, x):
        # TODO: encoder should return mu(mean) and variance
        # Try to use non-linear functions (e.g. relu) to improve your model
        pass

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        # TODO: using sampled z, return size 784 output 
        # Try to use non-linear functions (e.g. relu) to improve your model
        pass

    def forward(self, x):
        # TODO: The model should return flattened generated image (784), mu(mean), and variance
        # Think about VAE model architecture
        # x (img) --[encoder]-> mu, var --> sample z --[decoder]-> x' (generated img)
        pass