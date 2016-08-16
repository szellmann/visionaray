// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ALLOCATOR_H
#define VSNRAY_ALLOCATOR_H 1

#include "macros.h"

#include <cstddef>
#include <new>

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
    typedef T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

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

    pointer address(reference r) const
    {
        return &r;
    }

    const_pointer address(const_reference r) const
    {
        return &r;
    }

    pointer allocate(size_type n, void* /* hint */ = 0)
    {
        return (pointer)_mm_malloc(n * sizeof(T), A);
    }

    void deallocate(pointer p, size_type /* n */)
    {
        _mm_free(p);
    }

    size_t max_size() const
    {
        return static_cast<size_t>(-1) / sizeof(T);
    }

    void construct(pointer p, const_reference val)
    {
        new(static_cast<void*>(p)) T(val);
    }

    void destroy(pointer p)
    {
        p->T::~T();
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

#endif // VSNRAY_ALLOCATOR_H
