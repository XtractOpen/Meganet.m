# Meganet.m
A fresh approach to deep learning written in MATLAB

## Reporting Bugs

We are just getting started, so please be patient with us. If you find a bug,
 please report it by opening an issue or email lruthotto@emory.edu. In any case,
 include a small example that helps us re-produce the error. 
We'll work on this as quickly as possible.

## Getting started

1. Clone or download the code 
1. Add folder to your MATLAB path
1. (optional) run KernelTypes/mexcuda/make_cuda.m for fast CNNs using CuDNN
1. (optional) gather test data or binary files 

## Optional Binary Files

The `convMCN` kernel type and the average pooling require compiled binaries 
from the MatConvNet package. Please follow these [instructions](http://www.vlfeat.org/matconvnet/install/)
and add the files for `vl_nnconv`, `vl_nnconvt`, and `vl_nnpool` to your MATLAB path.

For best performance these files can be compiled with GPU or CuDNN support. 

## Additional Test Data

Some examples use these benchmark data

1. MNIST 
1. CIFAR10 
1. STL-10

## References 

The implementation is based on the ideas presented in:

1. Haber E, Ruthotto L: [Stable Architectures for Deep Neural Networks](http://arxiv.org/abs/1705.03341), Inverse Problems, 2017
1. Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: [Reversible Architectures for Arbitrarily Deep Residual Neural Networks](https://arxiv.org/abs/1709.03698), AAAI Conference on Artificial Intelligence 2018
1. Haber E, Ruthotto L, Holtham E, Jun SH:  [Learning across scales - A multiscale method for Convolution Neural Networks](https://arxiv.org/abs/1703.02009), AAAI Conference on Artificial Intelligence 2018

