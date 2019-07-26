import torch
from torch.nn import functional as F


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

    return BCE, KLD
