# resnet-cuda
RESNET CNN architecture enhanced by CUDA/C. Python interface for use with PyTorch.

Compiled for CUDA 12.6, CC 6.1 with cuDNN v8.9.7

src: CUDA source code for Resnet library

cu_resnet: Compiled library itself

comparison_time_test.py: Test/demonstration to compare training and inference time between a Resnet34 model trained on MNIST using a PyTorch backend and the custom cuda kernel backend. Demonstrates how to use custom cu_resnet to build models in PyTorch and backprop using PyTorch autograd. 
