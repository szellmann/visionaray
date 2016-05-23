// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifdef VSNRAY_HAVE_CUDA

#ifdef _WIN32
#include <windows.h> // APIENTRY
#endif
#include <cuda_gl_interop.h>

#include <visionaray/cuda/graphics_resource.h>

namespace visionaray
{
namespace cuda
{

cudaError_t graphics_resource::register_buffer(unsigned buffer, cudaGraphicsRegisterFlags flags)
{
    unregister();
    return cudaGraphicsGLRegisterBuffer(&resource_, buffer, flags);
}

cudaError_t graphics_resource::register_image(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags)
{
    unregister();
    return cudaGraphicsGLRegisterImage(&resource_, image, target, flags);
}

cudaError_t graphics_resource::unregister()
{
    if (resource_ == 0)
    {
        return cudaSuccess;
    }

    auto result = cudaGraphicsUnregisterResource(resource_);
    resource_ = 0;
    return result;
}

void* graphics_resource::map(size_t* size)
{
    auto err = cudaGraphicsMapResources(1, &resource_);
    if (err != cudaSuccess)
    {
        return 0;
    }

    err = cudaGraphicsResourceGetMappedPointer(&dev_ptr_, size, resource_);
    if (err != cudaSuccess)
    {
        cudaGraphicsUnmapResources(1, &resource_);
        dev_ptr_ = 0;
    }

    return dev_ptr_;
}

void* graphics_resource::map()
{
    size_t size = 0;
    return map(&size);
}

void graphics_resource::unmap()
{
    if (dev_ptr_ == 0)
    {
        return;
    }

    cudaGraphicsUnmapResources(1, &resource_);
    dev_ptr_ = 0;
}

} // cuda
} // visionaray

#endif // VSNRAY_HAVE_CUDA
