// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_GRAPHICS_RESOURCE_H
#define VSNRAY_CUDA_GRAPHICS_RESOURCE_H 1

#include <cuda_runtime_api.h>

#include <visionaray/detail/macros.h>

namespace visionaray
{
namespace cuda
{

class graphics_resource
{
public:

    graphics_resource();
   ~graphics_resource();

    cudaGraphicsResource_t get() const;

    cudaError_t register_buffer(unsigned buffer, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    cudaError_t register_image(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    cudaError_t unregister();

    void* map(size_t* size);
    void* map();
    void unmap();

    void* dev_ptr() const;

private:

    VSNRAY_NOT_COPYABLE(graphics_resource)

    cudaGraphicsResource_t resource_;
    void* dev_ptr_;

};

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_GRAPHICS_RESOURCE_H
