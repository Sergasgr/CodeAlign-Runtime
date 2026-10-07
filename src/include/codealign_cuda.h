#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace codealign {
    inline void check_cuda(cudaError_t err, const char* what, const char* file, int line) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " (" + what + ") at " + file + ":" + std::to_string(line));
        }
    }

    inline bool sync_check_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("CODEALIGN_SYNC_CHECK");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
}

#define CODEALIGN_CHECK_CUDA(call) ::codealign::check_cuda((call), #call, __FILE__, __LINE__)

#ifdef __CUDACC__
namespace codealign {
    template <typename Kernel, typename... Args>
    inline void launch(const char* name, const char* file, int line, Kernel kernel, dim3 grid, dim3 block, size_t shared_mem_bytes, cudaStream_t stream, Args&&... args) {
        kernel<<<grid, block, shared_mem_bytes, stream>>>(std::forward<Args>(args)...);
        check_cuda(cudaGetLastError(), name, file, line);
        if (sync_check_enabled()) check_cuda(cudaStreamSynchronize(stream), name, file, line);
    }
}  
#define CODEALIGN_KERNEL_LAUNCH(kernel, grid, block, shared_mem_bytes, stream, ...) ::codealign::launch(#kernel, __FILE__, __LINE__, kernel, grid, block, shared_mem_bytes, stream, __VA_ARGS__)
#endif
