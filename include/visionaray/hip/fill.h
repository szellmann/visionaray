// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_FILL_H
#define VSNRAY_HIP_FILL_H 1

#include <cstddef>

#include <hip/hip_runtime_api.h>

#include "safe_call.h"

namespace visionaray
{
namespace hip
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
    HIP_SAFE_CALL(hipMemcpy(ptr, host_array, len * count, hipMemcpyHostToDevice));
    delete[] host_array;
}

} // hip
} // visionaray

#endif // VSNRAY_HIP_FILL_H
