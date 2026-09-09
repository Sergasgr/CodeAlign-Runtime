from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='codealign_runtime_kernels',
    ext_modules=[
        CUDAExtension(
            name='codealign_runtime_kernels',
            sources=[
                'src/binding.cpp',                # Pybind11 bridge (gemv_int4_forward + flash_decoding_forward)
                'src/gemv_quantized.cu',          # Level 3 kernels (binding.cpp calls these)
                'src/flash_decoding_partial.cu',  # Level 4 partial kernel
                'src/flash_decoding_final.cu'     # Level 4 final kernel
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)