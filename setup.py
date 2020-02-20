from glob import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch_dwconv',
    version='0.1.0',
    description='Faster depthwise convolutions for PyTorch',
    url='https://github.com/rosinality/depthwise-conv-pytorch',
    author='Kim Seonghyeon',
    author_email='kim.seonghyeon@navercorp.com',
    license='MIT',
    python_requires='>=3.6',
    ext_modules=[
        CUDAExtension(
            'torch_dwconv._C',
            ['torch_dwconv/dwconv.cpp', 'torch_dwconv/dwconv_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
)
