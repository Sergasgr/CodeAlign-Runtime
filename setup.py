from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='codealign_runtime_kernels',
    ext_modules=[
        CUDAExtension(
            name='codealign_runtime_kernels',
            sources=[
                'src/binding.cpp', 
                'src/gemv_quantized.cu',
                # 'src/gemv_naive.cu',  # Descomentar esto si quiero exponerlos todos luego
                # 'src/gemv_optimized.cu'
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)