// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once 

#ifndef VSNRAY_CUDA_GRAPHICS_RESOURCE_H
#define VSNRAY_CUDA_GRAPHICS_RESOURCE_H

#include <cuda_runtime_api.h>

#include <visionaray/detail/macros.h>

namespace visionaray
{
namespace cuda
{

class graphics_resource
{
public:

    graphics_resource()
        : resource_(0)
        , dev_ptr_(0)
    {
    }

   ~graphics_resource()
    {
        unregister();
    }

    cudaGraphicsResource_t get() const { return resource_; }

    cudaError_t register_buffer(unsigned buffer, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    cudaError_t register_image(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    cudaError_t unregister();

    void* map(size_t* size);
    void* map();
    void unmap();

    void* dev_ptr() const { return dev_ptr_; }

private:

    VSNRAY_NOT_COPYABLE(graphics_resource)

    cudaGraphicsResource_t resource_;
    void* dev_ptr_;

};

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_GRAPHICS_RESOURCE_H
