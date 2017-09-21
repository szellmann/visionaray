// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_MANAGED_ALLOCATOR_H
#define VSNRAY_CUDA_MANAGED_ALLOCATOR_H 1

#include <cstddef>

#include <cuda_runtime_api.h>

namespace visionaray
{
namespace cuda
{

template <typename T>
class managed_allocator
{
public:

    typedef T value_type;

    managed_allocator() = default;

    template <typename U>
    managed_allocator(managed_allocator<U> const&) {}

    T* allocate(size_t n)
    {
        T* ptr = nullptr;
        cudaMallocManaged(&ptr, n * sizeof(T));
        return ptr;
    }

    void deallocate(T* ptr, size_t /* n */)
    {
        cudaFree(ptr);
    }

    bool operator==(managed_allocator const& /* rhs */) const
    {
        return true;
    }

    bool operator!=(managed_allocator const& rhs) const
    {
        return !(*this == rhs);
    }
};

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_MANAGED_ALLOCATOR_H
