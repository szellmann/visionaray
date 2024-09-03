// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_SAFE_CALL_H
#define VSNRAY_CUDA_SAFE_CALL_H 1

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime_api.h>

//-------------------------------------------------------------------------------------------------
// Macro to wrap cuda calls
//

#ifndef NDEBUG
#define CUDA_SAFE_CALL(FUNC) { visionaray::cuda::safe_call((FUNC), __FILE__, __LINE__); }
#else
#define CUDA_SAFE_CALL(FUNC) FUNC
#endif
#define CUDA_SAFE_CALL_X(FUNC) { visionaray::cuda::safe_call((FUNC), __FILE__, __LINE__, true); }


//-------------------------------------------------------------------------------------------------
// Macro to print last cuda error except if it was `Success'
//

#define CUDA_PRINT_LAST_ERROR() \
{ visionaray::cuda::print_last_error(cudaGetLastError(), __FILE__, __LINE__); }

namespace visionaray
{
namespace cuda
{

inline void safe_call(cudaError_t code, char const* file, int line, bool fatal = false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
        if (fatal)
        {
            exit(code);
        }
    }
}

inline void print_last_error(char const* file, int line)
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        fprintf(stderr, "LAST CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
    }
}

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_SAFE_CALL_H
