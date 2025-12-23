# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cu_resnet',  # Name of the package when imported
    version='0.1',
    description='High-performance fused ResNet ops with cuDNN+cuBLAS.',
    ext_modules=[
        CUDAExtension(
            name='cu_resnet._C',  # Imported as cu_resnet._C in Python
            sources=[
                'bindings.cu',
                'resnet_forward.cu',
                'resnet_backward.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=['cu_resnet'],
    package_dir={'cu_resnet': '.'}
)
