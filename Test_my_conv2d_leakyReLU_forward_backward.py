'''
Online lecture: Basics of PyTorch autograd
Demonstrate custom implementation of forward and backward propagation of a Conv2d-LeakyReLU neural network
'''
import torch
from torch import nn
from torch.nn.parameter import Parameter
from my_conv2d_v1 import MyConv2d_v1
from my_conv2d_v2 import MyConv2d_v2
from myLeakyReLU import MyLeakyReLU


flag_custom_implement_v1 = 1
flag_custom_implement_v2 = 1
flag_torch_built_ins_as_ref = 1

def main():
    # prepare the inputs and target data
    nSamples = 9
    inCh, outCh = 3, 2
    inImgRows, inImgCols = 16, 32
    knRows, knCols = 4, 5
    padding = 1
    stride = 2

    torch.manual_seed(1234)
    inputX = torch.randn(nSamples, inCh, inImgRows, inImgCols, requires_grad=True, dtype=torch.float64)
    input_weight = torch.randn(outCh, inCh, knRows, knCols, requires_grad=True, dtype=torch.float64)
    input_bias = torch.randn(outCh, requires_grad=True, dtype=torch.float64)


    '''
    Test #1: Test custom implementation #1 of forward and backward passed of 
    Conv2d_v1 and LeakyReLU using torch tensors
    '''
    if flag_custom_implement_v1:
        x1 = inputX.detach().clone()
        x1.requires_grad = True
        weight1 = input_weight.detach().clone()
        weight1.requires_grad = True
        bias1 = input_bias.detach().clone()
        bias1.requires_grad = True

        # forward pass
        # apply custom Conv2d_v1
        y1 = MyConv2d_v1.apply(x1, weight1, bias1, (padding, stride))
        y1.retain_grad()
        # apply custom LeakyRelu activation
        out1 = MyLeakyReLU.apply(y1, 0.02)
        out1.retain_grad()

        # Calculat loss: simply the mean of out1
        loss_1 = out1.mean()

        # backward pass including custom backward functions of MyConv2d and MyLeakyReLU
        loss_1.backward()

        print('Results of Test #1 (custom implementation of autograd using torch tensors):')
        print(x1.grad)
        print(weight1.grad)
        print(bias1.grad)

    '''
    Test #2: Test custom implementation #2 of forward and backward passed of 
    Conv2d_v2 and LeakyReLU using torch tensors
    '''
    if flag_custom_implement_v2:
        x2 = inputX.detach().clone()
        x2.requires_grad = True
        weight2 = input_weight.detach().clone()
        weight2.requires_grad = True
        bias2 = input_bias.detach().clone()
        bias2.requires_grad = True

        # forward pass
        # apply custom Conv2d_v2
        y2 = MyConv2d_v2.apply(x2, weight2, bias2, (padding, stride))
        y2.retain_grad()
        # apply custom LeakyRelu activation
        out2 = MyLeakyReLU.apply(y2, 0.02)
        out2.retain_grad()

        # Calculat loss: simply the mean of out2
        loss_2 = out2.mean()

        # backward pass including custom backward functions of MyConv2d and MyLeakyReLU
        loss_2.backward()

        print('Results of Test #2 (custom implementation of autograd using torch tensors):')
        print(x2.grad)
        print(weight2.grad)
        print(bias2.grad)


    '''
    Reference Test: : using the built-in torch.nn.conv2d, torch.LeakyReLU
    '''
    if flag_torch_built_ins_as_ref:
        # get the same input and parameters as above
        x3 = inputX.detach().clone()
        x3.requires_grad = True
        # using torch.nn.Conv2d as default
        torchConv2d = nn.Conv2d(inCh, outCh, kernel_size=(knRows, knCols), stride=stride, padding=padding)
        # set the trainable parameters of the conv2d module to the same weight and bias as used above
        torchConv2d.weight = Parameter(input_weight)
        torchConv2d.bias = Parameter(input_bias)

        # forward propagation
        y3 = torchConv2d(x3)
        y3.retain_grad()
        out3 = nn.LeakyReLU(0.02)(y3)
        out3.retain_grad()

        # Calculate MSE loss
        loss_mse_3 = out3.mean()

        # backward propagation
        loss_mse_3.backward()

        print('Results of Reference Test #3 (using Torch built-in conv2d and leakyReLu):')
        print(x3.grad)
        print(torchConv2d.weight.grad)
        print(torchConv2d.bias.grad)

    # Compare custom implementation #1 to the reference
    diff_grad_X_1_3 = x1.grad - x3.grad
    diff_grad_w_1_3 = weight1.grad - torchConv2d.weight.grad
    diff_grad_b_1_3 = bias1.grad - torchConv2d.bias.grad
    # clean up the infinite small number due to floating point error
    diff_grad_X_1_3[diff_grad_X_1_3 < 1e-14] = 0.0
    diff_grad_w_1_3[diff_grad_w_1_3 < 1e-14] = 0.0
    diff_grad_b_1_3[diff_grad_b_1_3 < 1e-14] = 0.0

    print('\nDifference between Test #1 and Reference Test results:')
    print('diff_grad_X_1_3 max difference:', diff_grad_X_1_3.abs().max().detach().numpy())
    print('diff_grad_w_1_3:', diff_grad_w_1_3.detach().numpy())
    print('diff_grad_b_1_3:', diff_grad_b_1_3.detach().numpy())

    # Compare custom implementation #2 to the reference
    diff_grad_X_2_3 = x2.grad - x3.grad
    diff_grad_w_2_3 = weight2.grad - torchConv2d.weight.grad
    diff_grad_b_2_3 = bias2.grad - torchConv2d.bias.grad
    # clean up the infinite small number due to floating point error
    diff_grad_X_2_3[diff_grad_X_2_3 < 1e-14] = 0.0
    diff_grad_w_2_3[diff_grad_w_2_3 < 1e-14] = 0.0
    diff_grad_b_2_3[diff_grad_b_2_3 < 1e-14] = 0.0

    print('\nDifference between Test #2 result and Reference Test result:')
    print('diff_grad_X_2_3 max difference:', diff_grad_X_2_3.abs().max().detach().numpy())
    print('diff_grad_w_2_3:', diff_grad_w_2_3.detach().numpy())
    print('diff_grad_b_2_3:', diff_grad_b_2_3.detach().numpy())

    print('Done!')


if __name__=='__main__':
    main()