from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='codealign_runtime_gemm',
    ext_modules=[
        CUDAExtension(
            name='codealign_runtime_gemm',
            sources=[
                'benchmarks/gemm_binding.cpp', 
                'src/gemm/gemm_quantized.cu'
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
