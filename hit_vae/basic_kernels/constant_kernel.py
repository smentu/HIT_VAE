import gpytorch
import torch

class ConstantKernel(gpytorch.kernels.Kernel):
    """
    Gives a covariance of one for all inputs. Should be combined with a scale kernel.
    """

    def __init__(self, **kwargs):
        super(ConstantKernel, self).__init__(has_lengthscale=False, **kwargs)

    def forward(self, x1, x2, diag=False, device='cuda:0', **params):
        return torch.ones(x1.shape[0], x2.shape[0]).to(device)
