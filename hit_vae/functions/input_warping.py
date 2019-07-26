import torch


class InputWarpingCovariance(torch.autograd.Function):
    """
    This pytorch autograd function is an input warping covariance function with gradients with respect to all inputs.
    This is meant to enable both parameter inference and label inference in models utilizing the input warping kernel.
    """

    # helper methods
    @staticmethod
    def iw(x, c, a, b):
        # basic input warping function as described in LonGP
        return 2 * c * (-0.5 + torch.div(1, 1 + torch.exp(-a * (x - b))))

    @staticmethod
    def td_scalegrad(x, c, a, b):
        # helper function for computing scale parameter gradient
        res = 2 * (-0.5 + torch.div(1, 1 + torch.exp(-a * (x - b))))
        return res

    @staticmethod
    def td_wsgrad(x, c, a, b):
        # helper function for computing window size parameter gradient
        res = 2 * c * (b - x) * torch.exp(-a * (x - b)) / (torch.exp(-a * (x - b)) + 1) ** 2
        return res

    @staticmethod
    def covar(x1, x2, lengthscale, scale, window_size, inflection_point):
        X1, X2 = torch.meshgrid(x1, x2)
        res = torch.exp((-(InputWarpingCovariance.iw(X1, scale, window_size, inflection_point)
                           - InputWarpingCovariance.iw(X2, scale, window_size, inflection_point)) ** 2)
                        / (2 * lengthscale ** 2))
        return res

    # PyTorch autograd functions comprise a forward and backward method
    @staticmethod
    def forward(ctx, x1, x2, l, c, a, b):
        if any(ctx.needs_input_grad[:2]):
            raise NotImplementedError()

        ctx.save_for_backward(x1, x2, l, c, a, b)
        return InputWarpingCovariance.covar(x1, x2, l, c, a, b)

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, l, c, a, b = ctx.saved_tensors
        grad_x1 = grad_x2 = grad_lengthscale = grad_scale = grad_window_size = grad_inflection_point = None
        X1, X2 = torch.meshgrid(x1, x2)

        if ctx.needs_input_grad[0]:
            raise NotImplementedError()

        if ctx.needs_input_grad[1]:
            raise NotImplementedError()

        if ctx.needs_input_grad[2]:
            raise NotImplementedError()

        if ctx.needs_input_grad[3]:
            grad_scale = InputWarpingCovariance.covar(x1, x2, l, c, a, b) \
                         * (InputWarpingCovariance.iw(X1, c, a, b) - InputWarpingCovariance.iw(X2, c, a, b)) \
                         * (InputWarpingCovariance.td_scalegrad(X1, c, a, b) - InputWarpingCovariance.td_scalegrad(X2,
                                                                                                                   c, a,
                                                                                                                   b))
            grad_scale = grad_scale.t().mm(grad_output)

        if ctx.needs_input_grad[4]:
            grad_window_size = InputWarpingCovariance.covar(x1, x2, l, c, a, b) \
                               * (InputWarpingCovariance.iw(X1, c, a, b) - InputWarpingCovariance.iw(X2, c, a, b)) \
                               * (InputWarpingCovariance.td_wsgrad(X1, c, a, b) - InputWarpingCovariance.td_wsgrad(X2,
                                                                                                                   c, a,
                                                                                                                   b))
            grad_window_size = grad_window_size.t().mm(grad_output)


        if ctx.needs_input_grad[5]:
            raise NotImplementedError()

        return grad_x1, grad_x2, grad_lengthscale, grad_scale, grad_window_size, grad_inflection_point
