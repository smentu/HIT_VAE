import torch
from torch.nn import functional as F
from torch import nn


class DigitVae(nn.Module):
    """
    Simple convolutional variational autoencoder to be used with MNIST images.
    """
    def __init__(self, ldim):
        super(DigitVae, self).__init__()

        # encoder network layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        # mean and logvar
        self.fc21 = nn.Linear(32, ldim)
        self.fc22 = nn.Linear(32, ldim)

        # decoder network layers
        self.fc3 = nn.Linear(ldim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, 16 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        # apply pooling between convolutional layers
        z = self.conv1(x)
        z = self.pool1(z)

        z = self.conv2(z)
        z = self.pool2(z)

        z = z.view(-1, 16 * 7 * 7)

        z = F.relu(self.fc1(z))

        return self.fc21(z), self.fc22(z)

    def sample_latent(self, mu, logvar):
        # print(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        x = x.view(-1, 16, 7, 7)

        # print(x.shape)
        x = self.deconv1(x)
        # print(x.shape)
        x = torch.sigmoid(self.deconv2(x))

        # print(x.shape)

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_latent(mu, logvar)
        # print(mu)
        return self.decode(z), mu, logvar
