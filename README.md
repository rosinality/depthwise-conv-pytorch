# depthwise-conv-pytorch
Faster depthwise convolutions for PyTorch

This implementation consists of 3 kernels from:

1. UpFirDn2D for large feature maps from StyleGAN2 (https://github.com/NVlabs/stylegan2)
2. DepthwiseConv2d for small feature maps from TensorFlow (https://github.com/tensorflow/tensorflow) and MXNet (https://github.com/apache/incubator-mxnet)
3. Backward filter kernels from PaddlePaddle (https://github.com/PaddlePaddle/Paddle)

I found this implementation faster than PyTorch native depthwise conv2d about 3~5x for larger feature maps, 1.5~2x for small feature maps if kernel size > 3. If used in EfficientNet, I got about 15% forward time speed ups.


## Installation

```bash
> python setup.py install
```


## Usage

Interface of depthwise_conv2d is same as F.conv2d or Conv2d. But currently this does not supports bis, dilations > 1, non-zero paddings, and in_channels != out_channels, that is, group multiplier > 1.

```python
import torch

from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

x = torch.randn(3, 5, 7, 9)
k = torch.randn(5, 1, 3, 3)

y = depthwise_conv2d(x, k, padding=1)

dwconv = DepthwiseConv2d(5, 5, padding=1)

y = dwconv(x)
```