// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#if VSNRAY_HAVE_CUDA

#ifdef _WIN32
#include <windows.h> // APIENTRY
#endif
#include <cuda_gl_interop.h>

#include <visionaray/cuda/graphics_resource.h>

namespace visionaray
{
namespace cuda
{

graphics_resource::graphics_resource()
    : resource_(0)
    , dev_ptr_(nullptr)
{
}

graphics_resource::~graphics_resource()
{
    unregister();
}

cudaGraphicsResource_t graphics_resource::get() const
{
    return resource_;
}

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
        dev_ptr_ = nullptr;
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
    if (dev_ptr_ == nullptr)
    {
        return;
    }

    cudaGraphicsUnmapResources(1, &resource_);
    dev_ptr_ = nullptr;
}

void* graphics_resource::dev_ptr() const
{
    return dev_ptr_;
}

} // cuda
} // visionaray

#endif // VSNRAY_HAVE_CUDA
