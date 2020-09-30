'''
Online lecture: Basics of PyTorch autograd
Demonstrate custom implementation #1 of forward and backward propagation of LeakyReLU
'''
import torch
from torch.autograd import Function


class MyLeakyReLU(Function):
    """
    Implement our own custom autograd Functions of LeakyReLU by subclassing
    torch.autograd.Function and overrdie the forward and backward passes
    """
    @staticmethod
    def forward(ctx, inX, negative_slope=0.01):
        """ override the forward function """
        ctx.negativ_slope = negative_slope
        out = inX.clone()
        out[inX < 0] *= negative_slope
        ctx.save_for_backward(inX)
        return out

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """ override backward function f """
        print('Performing custom backward of MyLeakyReLU')
        inX, = ctx.saved_tensors
        grad_X = grad_from_upstream.clone()
        grad_X[inX < 0] *= ctx.negativ_slope
        return grad_X, None



