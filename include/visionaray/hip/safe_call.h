// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_SAFE_CALL_H
#define VSNRAY_HIP_SAFE_CALL_H 1

#include <cstdio>

#include <hip/hip_runtime_api.h>

//-------------------------------------------------------------------------------------------------
// Macro to wrap hip calls
//

#ifndef NDEBUG
#define HIP_SAFE_CALL(FUNC) { visionaray::hip::safe_call((FUNC), __FILE__, __LINE__); }
#else
#define HIP_SAFE_CALL(FUNC) FUNC
#endif
#define HIP_SAFE_CALL_X(FUNC) { visionaray::hip::safe_call((FUNC), __FILE__, __LINE__, true); }

namespace visionaray
{
namespace hip
{

inline void safe_call(hipError_t code, char const* file, int line, bool fatal = false)
{
    if (code != hipSuccess)
    {
        fprintf(stderr, "HIP error: %s %s:%i\n", hipGetErrorString(code), file, line);
        if (fatal)
        {
            exit(code);
        }
    }
}

} // hip
} // visionaray

#endif // VSNRAY_HIP_SAFE_CALL_H
