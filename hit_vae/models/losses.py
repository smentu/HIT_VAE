import torch
import os
from torch.utils import data
from PIL import Image
from torch.nn import functional as F


# we use the Torch Dataset class to handle the data
class Dataset(data.Dataset):
    def __init__(self, data_location, list_IDs, labels, transform):
        self.data_location = data_location
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        filename = '{}.png'.format(ID)

        filepath = os.path.join(self.data_location, filename)
        image = Image.open(filepath)
        X = self.transform(image)
        y = self.labels[ID]

        # print(y)

        return X, y


def recon_loss(recon_x, x):
    # Binary cross-entropy between the original images and the reconstructions
    return F.binary_cross_entropy(recon_x, x, reduction='sum')


def normal_KLD(mu, logvar):
    # KL divergence between the latent representations and a unit normal distribution
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(recon_x, x, mu, logvar):
    BCE = recon_loss(recon_x, x)
    KLD = normal_KLD(mu, logvar)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    return BCE, KLD
