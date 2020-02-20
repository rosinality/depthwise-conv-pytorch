import torch
from torch.nn import functional as F

from torch_dwconv import depthwise_conv2d


BATCH_SIZE = 32


def get_diff(N, C, H, W, kernel_size, stride, padding):
    x = torch.randn(N, C, H, W).to('cuda')
    k = torch.randn(C, 1, kernel_size, kernel_size).to('cuda')

    native = F.conv2d(x, k, stride=stride, padding=padding, groups=C)
    custom = depthwise_conv2d(x, k, stride=stride, padding=padding)

    return (native - custom).abs().max().item()


def test_forward_large_size():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 3, 1, 1) < 1e-8


def test_forward_large_size_large_kernel():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 5, 1, 2) < 1e-8


def test_forward_large_size_stride():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 3, 2, 1) < 1e-8


def test_forward_large_size_large_kernel_stride():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 5, 2, 2) < 1e-8


def test_forward_large_size_no_pad():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 3, 1, 0) < 1e-8


def test_forward_large_size_large_kernel_no_pad():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 5, 1, 0) < 1e-8


def test_forward_large_size_stride_no_pad():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 3, 2, 0) < 1e-8


def test_forward_large_size_large_kernel_stride_no_pad():
    assert get_diff(BATCH_SIZE, 32, 64, 64, 5, 2, 0) < 1e-8


def test_forward_large_size_odd():
    assert get_diff(BATCH_SIZE, 32, 63, 65, 3, 1, 1) < 1e-8


def test_forward_large_size_odd_large_kernel():
    assert get_diff(BATCH_SIZE, 32, 63, 65, 5, 1, 2) < 1e-8


def test_forward_large_size_odd_stride():
    assert get_diff(BATCH_SIZE, 32, 63, 65, 3, 2, 1) < 1e-8


def test_forward_large_size_odd_large_kernel_stride():
    assert get_diff(BATCH_SIZE, 32, 63, 65, 5, 2, 2) < 1e-8


def test_forward_small_size():
    assert get_diff(BATCH_SIZE, 128, 32, 32, 3, 1, 1) < 1e-8


def test_forward_small_size_large_kernel():
    assert get_diff(BATCH_SIZE, 128, 32, 32, 5, 1, 2) < 1e-8
