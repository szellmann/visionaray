// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_ALIGNED_ALLOCATOR_H
#define VSNRAY_DETAIL_ALIGNED_ALLOCATOR_H 1

#include <cstddef>
#include <new>

#include <visionaray/math/simd/intrinsics.h> // VSNRAY_ARCH

#include "macros.h"

#if VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#include <mm_malloc.h>
#else
#include <malloc.h>
#endif


namespace visionaray
{

template <typename T, size_t A>
class aligned_allocator
{
public:

    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef size_t size_type;

    aligned_allocator() = default;

    aligned_allocator(aligned_allocator const& /* rhs */)
    {
    }

    template <typename U>
    aligned_allocator(aligned_allocator<U, A> const& /* rhs */)
    {
    }

    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, A> other;
    };

    pointer allocate(size_type n, void* /* hint */ = 0)
    {
        return (pointer)_mm_malloc(n * sizeof(T), A);
    }

    void deallocate(pointer p, size_type /* n */)
    {
        _mm_free(p);
    }

    bool operator==(aligned_allocator const& /* rhs */) const
    {
        return true;
    }

    bool operator!=(aligned_allocator const& rhs) const
    {
        return !(*this == rhs);
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_ALIGNED_ALLOCATOR_H
