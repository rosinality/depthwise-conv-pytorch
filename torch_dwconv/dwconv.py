import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import torch_dwconv._C


def use_dwconv2d_small(input, weight, up, stride, padding):
    return torch_dwconv._C.use_dwconv2d_small(
        input, weight, up[0], up[1], stride[0], stride[1], padding[0], padding[1]
    )


class DepthwiseConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, stride, pad, small):
        ctx.stride = stride
        ctx.pad = pad

        ctx.save_for_backward(input, kernel)

        ctx.use_small = small

        _, _, in_h, in_w = input.shape

        if ctx.use_small:
            out = torch_dwconv._C.dwconv2d_small(
                input, kernel, 1, 1, *stride, *pad, True
            )

        else:
            out = torch_dwconv._C.dwconv2d(
                input, kernel, 1, 1, *stride, pad[0], pad[0], pad[1], pad[1], True
            )

        _, _, out_h, out_w = out.shape

        ctx.g_pad = (
            kernel.shape[2] - pad[0] - 1,
            in_h - out_h * stride[0] + pad[0],
            kernel.shape[3] - pad[1] - 1,
            in_w - out_w * stride[1] + pad[1],
        )

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors

        stride = ctx.stride
        pad = ctx.pad
        g_pad = ctx.g_pad

        if ctx.use_small:
            grad_input = torch_dwconv._C.dwconv2d_small(
                grad_output, kernel, *stride, 1, 1, g_pad[0], g_pad[2], False
            )

        else:
            grad_input = torch_dwconv._C.dwconv2d(
                grad_output, kernel, *stride, 1, 1, *g_pad, False
            )

        grad_kernel = torch_dwconv._C.dwconv2d_backward_kernel(
            input, grad_output, kernel, 1, 1, *stride, *pad
        )

        return grad_input, grad_kernel, None, None, None, None


depthwise_conv2d_fn = DepthwiseConv2dFunction.apply


def make_tuple(value, n_value):
    if not isinstance(value, (list, tuple)):
        return (value,) * n_value

    else:
        n_item = len(value)

        if n_item > n_value:
            raise ValueError(
                f'Number items does not match with requirements: {n_item}, expected: {n_value}'
            )

        if len(value) == n_value:
            return value

        return value * n_value


def check_options(
    in_channels, out_channels, dilation, groups, bias, padding_mode='zeros'
):
    dilation = make_tuple(dilation, 2)

    if dilation[0] > 1 or dilation[1] > 1:
        raise ValueError('DepthwiseConv2d does not support dilations > 1')

    if groups is not None and groups != in_channels:
        raise ValueError('DepthwiseConv2d does not support groups != in_channels')

    if in_channels != out_channels:
        raise ValueError('DepthwiseConv2d does not support in_channels != out_channels')

    if bias:
        raise ValueError('DepthwiseConv2d does not support bias')

    if padding_mode != 'zeros':
        raise ValueError('DepthwiseConv2d does not non-zero paddings')


def depthwise_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=None
):
    stride = make_tuple(stride, 2)
    padding = make_tuple(padding, 2)

    check_options(input.shape[1], weight.shape[0], dilation, groups, bias)

    small = use_dwconv2d_small(input, weight, (1, 1), stride, padding)

    if not small and input.shape[2] <= 32 and input.shape[3] <= 32:
        return F.conv2d(
            input, weight, stride=stride, padding=padding, groups=input.shape[1]
        )

    return depthwise_conv2d_fn(input, weight, stride, padding, small)


class DepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=None,
        bias=False,
        padding_mode='zeros',
        custom=True,
    ):
        super().__init__()

        check_options(in_channels, out_channels, dilation, groups, bias, padding_mode)

        self.stride = stride
        self.padding = padding
        self.in_channel = in_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, 1, kernel_size, kernel_size)
        )

        self.custom = custom

    def forward(self, input):
        if self.custom:
            out = depthwise_conv2d(input, self.weight, None, self.stride, self.padding)

        else:
            out = F.conv2d(
                input,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                groups=input.shape[1],
            )

        return out
