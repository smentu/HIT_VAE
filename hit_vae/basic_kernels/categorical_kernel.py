import torch
import gpytorch
from gpytorch.kernels import Kernel


class CategoricalKernel(Kernel):
    """
    Computes a covariance matrix based on the categorical kernel between inputs
    """

    def __init__(self, **kwargs):
        super(CategoricalKernel, self).__init__(has_lengthscale=False, **kwargs)

    def forward(self, x1, x2, diag=False, device='cuda:0', **params):
        diff = self.covar_dist(x1, x2, diag=diag, **params)

        diff = (diff == 0).type(torch.FloatTensor).to(device)

        return diff
