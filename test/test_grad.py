import torch
from torch.autograd import gradcheck

from torch_dwconv import depthwise_conv2d


BATCH_SIZE = 8


def make_tensor(N, C, H, W, kernel_size, input_grad=False, kernel_grad=False):
    x = torch.randn(N, C, H, W).double().to('cuda')
    k = torch.randn(C, 1, kernel_size, kernel_size).double().to('cuda')

    x.requires_grad = input_grad
    k.requires_grad = kernel_grad

    return x, k


def check_input_grad(N, C, H, W, kernel_size, stride, padding):
    x, k = make_tensor(N, C, H, W, kernel_size, input_grad=True)

    result = gradcheck(
        lambda x_i: depthwise_conv2d(x_i, k, stride=stride, padding=padding).sum(),
        x,
        raise_exception=False,
    )

    return result


def check_kernel_grad(N, C, H, W, kernel_size, stride, padding):
    x, k = make_tensor(N, C, H, W, kernel_size, kernel_grad=True)

    result = gradcheck(
        lambda k_i: depthwise_conv2d(x, k_i, stride=stride, padding=padding).sum(),
        k,
        raise_exception=False,
    )

    return result


def test_input_grad_large_size():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 3, 1, 1)


def test_input_grad_large_size_large_kernel():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 5, 1, 2)


def test_input_grad_large_size_stride():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 3, 2, 1)


def test_input_grad_large_size_large_kernel_stride():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 5, 2, 2)


def test_input_grad_large_size_no_pad():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 3, 1, 0)


def test_input_grad_large_size_large_kernel_no_pad():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 5, 1, 0)


def test_input_grad_large_size_stride_no_pad():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 3, 2, 0)


def test_input_grad_large_size_large_kernel_stride_no_pad():
    assert check_input_grad(BATCH_SIZE, 8, 34, 34, 5, 2, 0)


def test_input_grad_large_size_odd():
    assert check_input_grad(BATCH_SIZE, 8, 33, 35, 3, 1, 1)


def test_input_grad_large_size_odd_large_kernel():
    assert check_input_grad(BATCH_SIZE, 8, 33, 35, 5, 1, 2)


def test_input_grad_large_size_odd_stride():
    assert check_input_grad(BATCH_SIZE, 8, 33, 35, 3, 2, 1)


def test_input_grad_large_size_odd_large_kernel_stride():
    assert check_input_grad(BATCH_SIZE, 8, 33, 35, 5, 2, 2)


def test_kernel_grad_large_size():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 3, 1, 1)


def test_kernel_grad_large_size_large_kernel():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 5, 1, 2)


def test_kernel_grad_large_size_stride():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 3, 2, 1)


def test_kernel_grad_large_size_large_kernel_stride():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 5, 2, 2)


def test_kernel_grad_large_size_no_pad():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 3, 1, 0)


def test_kernel_grad_large_size_large_kernel_no_pad():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 5, 1, 0)


def test_kernel_grad_large_size_stride_no_pad():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 3, 2, 0)


def test_kernel_grad_large_size_large_kernel_stride_no_pad():
    assert check_kernel_grad(BATCH_SIZE, 8, 34, 34, 5, 2, 0)


def test_kernel_grad_large_size_odd():
    assert check_kernel_grad(BATCH_SIZE, 8, 33, 35, 3, 1, 1)


def test_kernel_grad_large_size_odd_large_kernel():
    assert check_kernel_grad(BATCH_SIZE, 8, 33, 35, 5, 1, 2)


def test_kernel_grad_large_size_odd_stride():
    assert check_kernel_grad(BATCH_SIZE, 8, 33, 35, 3, 2, 1)


def test_kernel_grad_large_size_odd_large_kernel_stride():
    assert check_kernel_grad(BATCH_SIZE, 8, 33, 35, 5, 2, 2)


def test_input_grad_small_size():
    assert check_input_grad(BATCH_SIZE, 8, 16, 16, 3, 1, 1)


def test_input_grad_small_size_large_kernel():
    assert check_input_grad(BATCH_SIZE, 8, 16, 16, 5, 1, 2)


def test_kernel_grad_small_size():
    assert check_kernel_grad(BATCH_SIZE, 8, 16, 16, 3, 1, 1)


def test_kernel_grad_small_size_large_kernel():
    assert check_kernel_grad(BATCH_SIZE, 8, 16, 16, 5, 1, 2)
