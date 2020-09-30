'''
Online lecture: Basics of PyTorch autograd
Demonstrate custom implementation #1 of forward and backward propagation of Conv2d
'''
import torch
from torch.autograd import Function


class MyConv2d_v1(Function):
    """
    Verion #1 of our own custom autograd Functions of MyConv2d by subclassing
    torch.autograd.Function and overrdie the forward and backward passes
    """
    @staticmethod
    def forward(ctx, inX, in_weight, in_bias=None, convparam=None):
        """ override the forward function """
        # note: for demo purpose, assume dilation=1 and padding_mode='zeros',
        # also assume the padding and stride is the same for ROWS and COLS, respectively

        if convparam is not None:
            padding, stride = convparam
        else:
            padding, stride = 0, 1

        nOutCh, nInCh, nKnRows, nKnCols = in_weight.shape
        nImgSamples, nInCh, nInImgRows, nInImgCols = inX.shape

        # determine the output shape
        nOutRows = (nInImgRows + 2 * padding - nKnRows) // stride + 1
        nOutCols = (nInImgCols + 2 * padding - nKnCols) // stride + 1

        ''' 
        using torch.nn.functional.unfold to extract nL blocks of size of inChannels x nKnRows x nKnCols elements
        Each block can be used to do multiplication with the kernels
        Input shape:  (nImgSamples, nInCh, ∗)
        Output shape: (nImgSamples, nB = nInCh X ∏(kernel_sizes), nL = nOutRows X nOutCols)
        '''
        inX_nSamp_nB_nL = torch.nn.functional.unfold(inX, (nKnRows, nKnCols), padding=padding, stride=stride)
        inX_nSamp_nL_nB = inX_nSamp_nB_nL.transpose(1, 2)
        kn_nOutCh_nB = in_weight.view(nOutCh, -1)
        kn_nB_nOutCh = kn_nOutCh_nB.t()
        out_nSamp_nL_nOutCh = inX_nSamp_nL_nB.matmul(kn_nB_nOutCh)
        out_nSamp_nOutCh_nL = out_nSamp_nL_nOutCh.transpose(1, 2)
        out = out_nSamp_nOutCh_nL.reshape(nImgSamples, nOutCh, nOutRows, nOutCols)

        if in_bias is not None:
            out += in_bias.view(1, -1, 1, 1)

        # cache these objects for use in the backward pass
        ctx.InImgSize = (nInImgRows, nInImgCols)
        ctx.out_nSamp_nOutCh_nL_shape = out_nSamp_nOutCh_nL.shape
        ctx.inX_nSamp_nL_nB = inX_nSamp_nL_nB
        ctx.kn_nB_nOutCh = kn_nB_nOutCh
        ctx.parameters = (nOutCh, nInCh, nKnRows, nKnCols, padding, stride)
        # ctx.save_for_backward(inX_nSamp_nL_nB, kn_nB_nOutCh)

        return out

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """
        override the backward function. It receives a Tensor containing the gradient of the loss
        with respect to the output of the custom forward pass, and calculates the gradients of the loss
        with respect to each of the inputs of the custom forward pass.
        """
        grad_inputX = grad_weight = grad_bias = None

        print('Performing custom backward of MyConv2d')
        nOutCh, nInCh, nKnRows, nKnCols, padding, stride = ctx.parameters
        # inX_nSamp_nL_nB, kn_nB_nOutCh = ctx.saved_tensors

        # grad_out = torch.ones(out.shape, dtype=torch.float64) / out.numel()

        grad_bias = grad_from_upstream.sum(dim=[0, 2, 3])  # done for grad_bias

        grad_out_nSamp_nOutCh_nR_nC = grad_from_upstream

        # for: out_nSamp_nOutCh_nR_nC = out_nSamp_nOutCh_nL.reshape(nSamples, outCh, nOutRows, nOutCols)
        grad_out_nSamp_nOutCh_nL = grad_out_nSamp_nOutCh_nR_nC.reshape(ctx.out_nSamp_nOutCh_nL_shape)

        # for: out_nSamp_nOutCh_nL = out_nSamp_nL_nOutCh.transpose(1, 2)
        grad_out_nSamp_nL_nOutCh = grad_out_nSamp_nOutCh_nL.transpose(1, 2)

        # for: out_nSamp_nL_nOutCh = inX_nSamp_nL_nB.matmul(kn_nB_nOutCh)
        grad_inX_nSamp_nL_nB = grad_out_nSamp_nL_nOutCh.matmul(ctx.kn_nB_nOutCh.t())

        # continue to finish calculation of the gradient w.r.t "weight", i.e. the convolution kernel
        grad_kn_nB_nOutCh = ctx.inX_nSamp_nL_nB.transpose(1, 2).matmul(grad_out_nSamp_nL_nOutCh)
        grad_kn_nB_nOutCh = grad_kn_nB_nOutCh.sum(dim=0)
        grad_kn_nOutCh_nB = grad_kn_nB_nOutCh.t()
        grad_weight = grad_kn_nOutCh_nB.view(nOutCh, nInCh, nKnRows, nKnCols)  # done for grad_weight

        # for: inX_nSamp_nL_nB = inX_nSamp_nB_nL.transpose(1, 2)
        grad_inX_nSamp_nB_nL = grad_inX_nSamp_nL_nB.transpose(1, 2)

        # for: inX_nSamp_nB_nL = torch.nn.functional.unfold(inputX, (ctx.nKnRows, ctx.nKnCols))
        grad_inputX = torch.nn.functional.fold(grad_inX_nSamp_nB_nL, ctx.InImgSize, (nKnRows, nKnCols),
                                                padding=padding, stride=stride)

        return grad_inputX, grad_weight, grad_bias, None


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


