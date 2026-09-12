from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='codealign_runtime_transformer',
    ext_modules=[
        CUDAExtension(
            name='codealign_runtime_transformer',
            sources=[
                'src/transformer_binding.cpp',
                'src/transformer.cpp',
                'src/memory.cpp',
                'src/gemv_quantized.cu',
                'src/flash_decoding_partial.cu',
                'src/flash_decoding_final.cu',
                'src/rope.cu',                 
                'src/activations.cu',          
                'src/rmsnorm.cu',              
                'src/residual_ops.cu'
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)