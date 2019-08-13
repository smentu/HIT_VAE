import torch
from gpytorch.kernels import Kernel


# class InputWarpKernel(Kernel):
#
#     @staticmethod
#     def input_warp(x, c, a, b):
#         omega = 2 * c * (-0.5 + torch.div(1, (1 + torch.exp(-a * (x - b)))))
#         return omega
#
#     def __init__(self, **kwargs):
#         super(InputWarpKernel, self).__init__(has_lengthscale=True, **kwargs)
#
#     def forward(self, x1, x2, diag=False, **params):
#         # print(x1)
#         x1_ = InputWarpKernel.input_warp(x1)
#         # print(x1_)
#         x2_ = InputWarpKernel.input_warp(x2)
#
#         res = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
#
#         res = res.mul(-1 / self.lengthscale)
#
#         res = torch.exp(res)
#
#         return res


def input_warping(x, c, a, b):
    omega = 2 * c * (-0.5 + 1 / (1 + torch.exp(-a * (x - b))))
    return omega

class InputWarpingKernel(Kernel):
    """
    A simple version of the input warping kernel described in LonGP built on top of the GPYTorch covar dist
    functionality.
    """

    def __init__(self, c_prior=None, a_prior=None, b_prior=None, **kwargs):
        super(InputWarpingKernel, self).__init__(has_lengthscale=True, **kwargs)

        self.register_parameter(
            name="c",
            parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )

        self.register_parameter(
            name="a",
            parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )

        self.register_parameter(
            name="b",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

    def forward(self, x1, x2, diag=False, **params):
        # print(x1)
        x1_ = input_warping(x1, self.c, self.a, self.b)
        # print(x1_)
        x2_ = input_warping(x2, self.c, self.a, self.b)

        res = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)

        res = res.mul(-1 / self.lengthscale)

        res = torch.exp(res)

        return res