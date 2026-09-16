from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='codealign_runtime_kernels',
    ext_modules=[
        CUDAExtension(
            name='codealign_runtime_kernels',
            sources=[
                'benchmarks/gemv_binding.cpp',
                'src/gemv/gemv_quantized.cu',
                'src/ops/flash_decoding_partial.cu',
                'src/ops/flash_decoding_final.cu'
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)