// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

#include "cuda.h"

namespace visionaray
{
namespace cuda
{

cudaError_t init_gl_interop(int device)
{
    cudaError_t result = cudaSuccess;

    cudaDeviceProp prop;
    result = cudaChooseDevice(&device, &prop);
    if (result != cudaSuccess)
    {
        return result;
    }

    result = cudaGLSetGLDevice(device);

    if (result == cudaErrorSetOnActiveProcess)
    {
        result = cudaDeviceReset();
        result = cudaGLSetGLDevice(device);
    }

    return result;
}

} // cuda
} // visionaray
