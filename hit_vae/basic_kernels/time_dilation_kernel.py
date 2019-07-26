import torch
from gpytorch.kernels import Kernel


class TimeDilationKernel(Kernel):
    """
    A quick proof of concept version of the input warping kernel with all parameters set to constants.
    """

    @staticmethod
    def time_warp(t):
        omega = (-0.5 + 1 / (1 + torch.exp(-t)))
        return omega

    def __init__(self, **kwargs):
        super(TimeDilationKernel, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        # print(x1)
        x1_ = TimeDilationKernel.time_warp(x1)
        # print(x1_)
        x2_ = TimeDilationKernel.time_warp(x2)

        res = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)

        res = res.mul(-1 / self.lengthscale)

        return res
