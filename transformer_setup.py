from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='codealign_runtime_transformer',
    ext_modules=[
        CUDAExtension(
            name='codealign_runtime_transformer',
            sources=[
                'src/transformer/transformer_binding.cpp',
                'src/transformer/transformer.cpp',
                'src/transformer/memory.cpp',
                'src/gemv/gemv_quantized.cu',
                'src/gemm/gemm_naive.cu',
                'src/ops/flash_decoding_partial.cu',
                'src/ops/flash_decoding_final.cu',
                'src/ops/rope.cu',                 
                'src/ops/activations.cu',          
                'src/ops/rmsnorm.cu',              
                'src/ops/residual_ops.cu',
                'src/speculative/speculative.cpp'
            ],
            include_dirs=[os.path.abspath('src')],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)