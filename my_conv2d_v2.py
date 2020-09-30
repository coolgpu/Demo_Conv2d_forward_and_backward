'''
Online lecture: Basics of PyTorch autograd
Demonstrate custom implementation #2 of forward and backward propagation of Conv2d
'''
import torch
from torch.autograd import Function


class MyConv2d_v2(Function):
    """
    Version #2 of our own custom autograd Functions of MyConv2d by subclassing
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

        paddedX = torch.zeros((nImgSamples, nInCh, nInImgRows+2*padding, nInImgCols+2*padding), dtype=inX.dtype)
        paddedX[:,:,padding:nInImgRows+padding,padding:nInImgCols+padding] = inX

        # determine the output shape
        nOutRows = (nInImgRows + 2 * padding - nKnRows) // stride + 1
        nOutCols = (nInImgCols + 2 * padding - nKnCols) // stride + 1

        out = torch.zeros(nImgSamples, nOutCh, nOutRows, nOutCols, dtype=inX.dtype)
        for outCh in range(nOutCh):
            for iRow in range(nOutRows):
                startRow = iRow * stride
                for iCol in range(nOutCols):
                    startCol = iCol * stride
                    out[:, outCh, iRow, iCol] = \
                        (paddedX[:,:,startRow:startRow+nKnRows,startCol:startCol+nKnCols] \
                        * in_weight[outCh,:,0:nKnRows,0:nKnCols]).sum(axis=(1,2,3))

        if in_bias is not None:
            out += in_bias.view(1, -1, 1, 1)

        ctx.parameters = (padding, stride)
        ctx.save_for_backward(paddedX, in_weight, in_bias)

        return out

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """
        override the backward function. It receives a Tensor containing the gradient of the loss
        with respect to the output of the custom forward pass, and calculates the gradients of the loss
        with respect to each of the inputs of the custom forward pass.
        """
        print('Performing custom backward of MyConv2d_v2')
        padding, stride = ctx.parameters
        paddedX, in_weight, in_bias = ctx.saved_tensors
        nImgSamples, nInCh, nPadImgRows, nPadImgCols = paddedX.shape
        nOutCh, nInCh, nKnRows, nKnCols = in_weight.shape
        nImgSamples, nOutCh, nOutRows, nOutCols = grad_from_upstream.shape

        grad_padX = torch.zeros_like(paddedX)
        grad_weight = torch.zeros_like(in_weight)
        for outCh in range(nOutCh):
            for iRow in range(nOutRows):
                startRow = iRow * stride
                for iCol in range(nOutCols):
                    startCol = iCol * stride

                    grad_padX[:,:,startRow:startRow+nKnRows,startCol:startCol+nKnCols] += \
                        grad_from_upstream[:, outCh, iRow, iCol].reshape(-1, 1, 1, 1) * \
                        in_weight[outCh, :, 0:nKnRows, 0:nKnCols]

                    grad_weight[outCh, :, 0:nKnRows, 0:nKnCols] += \
                        (paddedX[:,:,startRow:startRow+nKnRows,startCol:startCol+nKnCols] * \
                        grad_from_upstream[:, outCh, iRow, iCol].reshape(-1, 1, 1, 1)).sum(axis=0)

        grad_inputX = grad_padX[:,:,padding:nPadImgRows-padding,padding:nPadImgCols-padding]

        if in_bias is not None:
            grad_bias = grad_from_upstream.sum(axis=(0, 2, 3))
        else:
            grad_bias = None

        return grad_inputX, grad_weight, grad_bias, None
