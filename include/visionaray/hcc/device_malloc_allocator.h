// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_DEVICE_MALLOC_ALLOCATOR_H
#define VSNRAY_HCC_DEVICE_MALLOC_ALLOCATOR_H 1

#include <cstddef>

#include <hcc/hc_am.hpp>

#include "../detail/macros.h"

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// Allocator class that allocates storage with hc::am_alloc()
//

template <typename T>
class device_malloc_allocator
{
public:

    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    VSNRAY_FUNC
    device_malloc_allocator() = default;

    VSNRAY_FUNC
    device_malloc_allocator(device_malloc_allocator const& /* rhs */)
    {
    }

    template <typename U>
    VSNRAY_FUNC
    device_malloc_allocator(device_malloc_allocator<U> const& /* rhs */)
    {
    }

    template <typename U>
    struct rebind
    {
        typedef device_malloc_allocator<U> other;
    };

    VSNRAY_FUNC
    pointer address(reference r) const
    {
        return &r;
    }

    VSNRAY_FUNC
    const_pointer address(const_reference r) const
    {
        return &r;
    }

    VSNRAY_CPU_FUNC
    pointer allocate(size_type n, void* hint = 0)
    {
        return (pointer)hc::am_alloc(
                n * sizeof(T), accelerator_,
                hint != 0 ? *reinterpret_cast<int*>(hint) : 0
                );
    }

    VSNRAY_CPU_FUNC
    void deallocate(pointer p, size_type /* n */)
    {
        hc::am_free(p);
    }

    size_t max_size() const
    {
        return static_cast<size_t>(-1) / sizeof(T);
    }

    VSNRAY_FUNC
    bool operator==(device_malloc_allocator const& /* rhs */) const
    {
        return true;
    }

    VSNRAY_FUNC
    bool operator!=(device_malloc_allocator const& rhs) const
    {
        return !(*this == rhs);
    }

private:

    hc::accelerator accelerator_;

};

} // hcc
} // visionaray

#endif // VSNRAY_HCC_DEVICE_MALLOC_ALLOCATOR_H
