// This file is distributed under the MIT license.
// See the LICENSE file for details.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef VSNRAY_HIP_MANAGED_ALLOCATOR_H
#define VSNRAY_HIP_MANAGED_ALLOCATOR_H 1

#include <cstddef>

#include <hip/hip_runtime_api.h>

namespace visionaray
{
namespace hip
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
        hipMallocManaged(&ptr, n * sizeof(T));
        return ptr;
    }

    void deallocate(T* ptr, size_t /* n */)
    {
        hipFree(ptr);
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

} // hip
} // visionaray

#endif // VSNRAY_HIP_MANAGED_ALLOCATOR_H
