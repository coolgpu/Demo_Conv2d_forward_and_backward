"""
Extra: Demonstrate application of Conv2d with pre-defined kernels for
Sobel edge detection and Gaussian blurring
"""
import torch
from torchvision.transforms import ToTensor
from my_conv2d_v1 import MyConv2d_v1
from PIL import Image
import numpy as np

flag_sobel_edge_detection = 1
flag_gaussian_blur = 1


def save_tensor_to_png(tensor, outfile):
    out_np = tensor.squeeze().detach().numpy()
    out_np_u8 = (out_np * 255.0/out_np.max()).astype(np.uint8)
    out_img = Image.fromarray(out_np_u8)
    out_img.save(outfile)


def main():
    file = r'leaf_512_768_original.png'
    img = Image.open(file)
    input_img_tensor = ToTensor()(img).unsqueeze(0)
    stride = 1

    ''' Sobel edge detection using Conv2d '''
    if flag_sobel_edge_detection:
        weight1 = torch.tensor([[1., 0., -1.], [2., 0., -2], [1., 0., -1.]]).view(1, 1, 3, 3)
        weight2 = torch.tensor([[1., 2.,  1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)
        padding = 1  # half padding

        y1 = MyConv2d_v1.apply(input_img_tensor, weight1, None, (padding, stride))
        y2 = MyConv2d_v1.apply(input_img_tensor, weight2, None, (padding, stride))
        y_sobel = torch.sqrt(y1 * y1 + y2 * y2)

        save_tensor_to_png(y_sobel, 'leaf_sobel_edge_out.png')

    ''' Gaussian blur using Conv2d '''
    if flag_gaussian_blur:
        weight_gauss = torch.tensor(
            [[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
             [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
             [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]).view(1, 1, 5, 5)
        padding = 2  # half padding

        y_gauss = MyConv2d_v1.apply(input_img_tensor, weight_gauss, None, (padding, stride))

        save_tensor_to_png(y_gauss, 'leaf_gaussian_blur_out.png')


if __name__ == '__main__':
    main()
