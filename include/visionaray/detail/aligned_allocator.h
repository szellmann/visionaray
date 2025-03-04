// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_ALIGNED_ALLOCATOR_H
#define VSNRAY_DETAIL_ALIGNED_ALLOCATOR_H 1

#include <cstddef>
#include <cstdlib>
#include <new>

#include "compiler.h"
#if VSNRAY_CXX_MSVC
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
#if VSNRAY_CXX_MSVC
        return (pointer)_mm_malloc(n * sizeof(T), A);
#else
        value_type* ptr{nullptr};
        auto ret = posix_memalign((void**)&ptr, A, sizeof(T) * n);
        if (ret != 0)
        {
            throw std::bad_alloc();
        }
        return ptr;
#endif
    }

    void deallocate(pointer p, size_type /* n */)
    {
#if VSNRAY_CXX_MSVC
        _mm_free(p);
#else
        std::free(p);
#endif
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
