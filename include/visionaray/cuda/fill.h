// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_FILL_H
#define VSNRAY_CUDA_FILL_H 1

#include <cstddef>

#include <cuda_runtime_api.h>

#include "safe_call.h"

namespace visionaray
{
namespace cuda
{

inline void fill(void* ptr, size_t len, void* bytes, unsigned count)
{
    char* host_array = new char[len * count];
    for (size_t i = 0; i < len; ++i)
    {
        for (unsigned c = 0; c < count; ++c)
        {
            host_array[i * count + c] = ((char*)bytes)[c];
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy(ptr, host_array, len * count, cudaMemcpyHostToDevice));
    delete[] host_array;
}

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_FILL_H
