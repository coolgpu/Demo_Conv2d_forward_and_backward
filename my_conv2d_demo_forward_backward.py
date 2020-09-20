'''
Demonstrate custom implementation of forward and backward propagation of a Conv2d-LeakyReLU neural network
'''
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.parameter import Parameter

# prepare the inputs and target data
nSamples = 5
inCh, outCh = 3, 2
inImgRows, inImgCols = 16, 32
knRows, knCols = 4, 5
padding = 1
stride = 1

torch.manual_seed(1234)
inputX = torch.randn(nSamples, inCh, inImgRows, inImgCols, requires_grad=True, dtype=torch.float64)
input_weight = torch.randn(outCh, inCh, knRows, knCols, requires_grad=True, dtype=torch.float64)
input_bias = torch.randn(outCh, requires_grad=True, dtype=torch.float64)


class MyConv2d(Function):
    """
    Implement our own custom autograd Functions of MyConv2d by subclassing
    torch.autograd.Function and overriding the forward and backward passes
    """
    @staticmethod
    def forward(ctx, inX, in_weight, in_bias=None, convparam=None):
        """
        override the forward function
        """
        # note: for demo purpose, assume dilation=1 and padding_mode='zeros',
        # also assume the padding and stride is the same for ROWS and COLS, respectively

        if convparam is not None:
            ctx.padding, ctx.stride = convparam
        else:
            ctx.padding, ctx.stride = 0, 1

        nOutCh, nInCh, nKnRows, nKnCols = in_weight.shape
        nImgSamples, nInCh, nInImgRows, nInImgCols = inX.shape

        # determine the output shape
        nOutRows = (nInImgRows + 2 * ctx.padding - knRows) // ctx.stride + 1
        nOutCols = (nInImgCols + 2 * ctx.padding - knCols) // ctx.stride + 1

        ''' 
        using torch.nn.functional.unfold to extract nL blocks of size of inChannels x knRows x knCols elements
        Each block can be used to do multiplication with the kernels
        Input shape:  (nImgSamples, nInCh, ∗)
        Output shape: (nImgSamples, nB = nInCh X ∏(kernel_sizes), nL = nOutRows X nOutCols)
        '''
        inX_nSamp_nB_nL = torch.nn.functional.unfold(inX, (nKnRows, nKnCols), padding=ctx.padding, stride=ctx.stride)
        inX_nSamp_nL_nB = inX_nSamp_nB_nL.transpose(1, 2)
        kn_nOutCh_nB = in_weight.view(nOutCh, -1)
        kn_nB_nOutCh = kn_nOutCh_nB.t()
        out_nSamp_nL_nOutCh = inX_nSamp_nL_nB.matmul(kn_nB_nOutCh)
        out_nSamp_nOutCh_nL = out_nSamp_nL_nOutCh.transpose(1, 2)
        out = out_nSamp_nOutCh_nL.reshape(nImgSamples, nOutCh, nOutRows, nOutCols)

        if in_bias is not NotImplementedError:
            out += in_bias.view(1, -1, 1, 1)

        # cache these objects for use in the backward pass
        ctx.InImgSize = (nInImgRows, nInImgCols)
        ctx.out_nSamp_nOutCh_nL_shape = out_nSamp_nOutCh_nL.shape
        ctx.inX_nSamp_nL_nB = inX_nSamp_nL_nB
        ctx.kn_nB_nOutCh = kn_nB_nOutCh
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
        # inX_nSamp_nL_nB, kn_nB_nOutCh = ctx.saved_tensors

        grad_bias = grad_from_upstream.sum(dim=[0, 2, 3])  # done for grad_bias

        grad_out_nSamp_nOutCh_nR_nC = grad_from_upstream

        # based on: out_nSamp_nOutCh_nR_nC = out_nSamp_nOutCh_nL.reshape(nSamples, outCh, nOutRows, nOutCols)
        grad_out_nSamp_nOutCh_nL = grad_out_nSamp_nOutCh_nR_nC.reshape(ctx.out_nSamp_nOutCh_nL_shape)

        # based on: out_nSamp_nOutCh_nL = out_nSamp_nL_nOutCh.transpose(1, 2)
        grad_out_nSamp_nL_nOutCh = grad_out_nSamp_nOutCh_nL.transpose(1, 2)

        # based on: out_nSamp_nL_nOutCh = inX_nSamp_nL_nB.matmul(kn_nB_nOutCh)
        grad_inX_nSamp_nL_nB = grad_out_nSamp_nL_nOutCh.matmul(ctx.kn_nB_nOutCh.t())

        # continue to finish calculation of the gradient w.r.t "weight", i.e. the convolution kernel
        grad_kn_nB_nOutCh = ctx.inX_nSamp_nL_nB.transpose(1, 2).matmul(grad_out_nSamp_nL_nOutCh)
        grad_kn_nB_nOutCh = grad_kn_nB_nOutCh.sum(dim=0)
        grad_kn_nOutCh_nB = grad_kn_nB_nOutCh.t()
        grad_weight = grad_kn_nOutCh_nB.view(outCh, inCh, knRows, knCols)  # done for grad_weight

        # based on: inX_nSamp_nL_nB = inX_nSamp_nB_nL.transpose(1, 2)
        grad_inX_nSamp_nB_nL = grad_inX_nSamp_nL_nB.transpose(1, 2)

        # based on: inX_nSamp_nB_nL = torch.nn.functional.unfold(inputX, (knRows, knCols))
        grad_inputX = torch.nn.functional.fold(grad_inX_nSamp_nB_nL, ctx.InImgSize, (knRows, knCols),
                                                padding=ctx.padding, stride=ctx.stride)

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

'''
Test #1: using our custom implementation of MyConv2d and MyLeakyReLU
'''
if 1:
    x1 = inputX.detach().clone()
    x1.requires_grad = True
    weight1 = input_weight.detach().clone()
    weight1.requires_grad = True
    bias1 = input_bias.detach().clone()
    bias1.requires_grad = True

    # forward pass
    # apply custom Conv2d
    y1 = MyConv2d.apply(x1, weight1, bias1, (padding, stride))
    y1.retain_grad()
    # apply custom LeakyRelu activation
    out1 = MyLeakyReLU.apply(y1, 0.02)
    out1.retain_grad()

    # Calculat loss simply as the mean of out1
    loss_1 = out1.mean()

    # backward pass including custom backward functions of MyLeakyReLU and MyConv2d
    loss_1.backward()

    # print('Results of Test #1 (custom implementation of MyLeakyReLU and MyConv2d):')
    # print(x1.grad)
    # print(weight1.grad)
    # print(bias1.grad)

'''
Test #2 (as a reference): using the built-in torch.nn.conv2d, torch.LeakyReLU
'''
if 1:
    # get the same input and parameters as above
    x2 = inputX.detach().clone()
    x2.requires_grad = True
    # using torch.nn.Conv2d as default
    torchConv2d = nn.Conv2d(inCh, outCh, kernel_size=(knRows, knCols), stride=stride, padding=padding)
    # set the trainable parameters of the conv2d module to the same weight and bias as used Test #1
    torchConv2d.weight = Parameter(input_weight)
    torchConv2d.bias = Parameter(input_bias)

    # forward propagation
    y2 = torchConv2d(x2)
    y2.retain_grad()
    out2 = nn.LeakyReLU(0.02)(y2)
    out2.retain_grad()

    # Calculate loss simply as the mean of out2
    loss_mse_2 = out2.mean()

    # backward propagation
    loss_mse_2.backward()

    # print('Results of Reference Test #2 (using Torch built-in conv2d and leakyReLu):')
    # print(x2.grad)
    # print(torchConv2d.weight.grad)
    # print(torchConv2d.bias.grad)

# Compare custom implementation using tensors (Test #1) to the reference (test #2)
diff_out1_out2  = out1 - out2
diff_grad_X_1_2 = x1.grad - x2.grad
diff_grad_w_1_2 = weight1.grad - torchConv2d.weight.grad
diff_grad_b_1_2 = bias1.grad - torchConv2d.bias.grad
# clean up the infinite small number due to floating point error
diff_out1_out2[diff_out1_out2 < 1e-12] = 0.0
diff_grad_X_1_2[diff_grad_X_1_2 < 1e-12] = 0.0
diff_grad_w_1_2[diff_grad_w_1_2 < 1e-12] = 0.0
diff_grad_b_1_2[diff_grad_b_1_2 < 1e-12] = 0.0

print('\nDifference between Test #1 and Reference Test #2')
print('diff_out1_out2 max difference:', diff_out1_out2.abs().max().detach().numpy())
print('diff_grad_X_1_2 max difference:', diff_grad_X_1_2.abs().max().detach().numpy())
print('diff_grad_w_1_2:', diff_grad_w_1_2.detach().numpy())
print('diff_grad_b_1_2:', diff_grad_b_1_2.detach().numpy())


print('All matched.')
