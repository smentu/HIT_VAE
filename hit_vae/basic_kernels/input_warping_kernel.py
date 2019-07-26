import torch
from gpytorch.kernels import Kernel


class InputWarpKernel(Kernel):
    """
    A simple version of the input warping kernel described in LonGP built on top of the GPYTorch covar dist
    functionality.
    """

    @staticmethod
    def input_warp(x, c, a, b):
        omega = 2 * c * (-0.5 + torch.div(1, (1 + torch.exp(-a * (x - b)))))
        return omega

    def __init__(self, **kwargs):
        super(InputWarpKernel, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        # print(x1)
        x1_ = InputWarpKernel.time_warp(x1)
        # print(x1_)
        x2_ = InputWarpKernel.time_warp(x2)

        res = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)

        res = res.mul(-1 / self.lengthscale)

        return res
