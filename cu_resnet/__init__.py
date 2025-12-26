# __init__.py
from .._C import (
    fused_conv_bn_relu_forward,
    fused_conv_bn_forward,
    fused_conv_bn_relu_manual_backward,
    fused_conv_bn_manual_backward,
    init_libs,
    destroy_libs,
)

# Auto-initialize the CUDA libraries when the package is imported
init_libs()
