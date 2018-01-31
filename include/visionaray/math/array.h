// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_ARRAY_H
#define VSNRAY_MATH_ARRAY_H 1

#include <cstddef>
#include <iterator>

#ifdef __CUDACC__
#include <thrust/iterator/reverse_iterator.h>
#endif

#include "../detail/macros.h"
#include "config.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// array
//
// Implements the functionality provided by std::array, but with all access functions
// declared as host/device for compatibility with NVIDIA nvcc
//

template <typename T, size_t N>
struct array
{
    using value_type                = T;
    using size_type                 = size_t;
    using difference_type           = ptrdiff_t;
    using reference                 = value_type&;
    using const_reference           = value_type const&;
    using pointer                   = T*;
    using const_pointer             = T const*;
    using iterator                  = T*;
    using const_iterator            = T const*;

#ifdef __CUDACC__
    using reverse_iterator          = thrust::reverse_iterator<iterator>;
    using const_reverse_iterator    = thrust::reverse_iterator<const_iterator>;
#else
    using reverse_iterator          = std::reverse_iterator<iterator>;
    using const_reverse_iterator    = std::reverse_iterator<const_iterator>;
#endif


    MATH_FUNC reference               at(size_type pos);
    MATH_FUNC const_reference         at(size_type pos) const;

    MATH_FUNC reference               operator[](size_type pos);
    MATH_FUNC const_reference         operator[](size_type pos) const;

    MATH_FUNC reference               front();
    MATH_FUNC const_reference         front() const;

    MATH_FUNC reference               back();
    MATH_FUNC const_reference         back() const;

    MATH_FUNC T*                      data();
    MATH_FUNC T const*                data() const;


    MATH_FUNC iterator                begin();
    MATH_FUNC const_iterator          begin() const;
    MATH_FUNC const_iterator          cbegin() const;

    MATH_FUNC iterator                end();
    MATH_FUNC const_iterator          end() const;
    MATH_FUNC const_iterator          cend() const;

    MATH_FUNC reverse_iterator        rbegin();
    MATH_FUNC const_reverse_iterator  rbegin() const;
    MATH_FUNC const_reverse_iterator  crbegin() const;

    MATH_FUNC reverse_iterator        rend();
    MATH_FUNC const_reverse_iterator  rend() const;
    MATH_FUNC const_reverse_iterator  crend() const;


#ifdef VSNRAY_CXX_HAS_CONSTEXPR
    MATH_FUNC constexpr bool          empty() const;
    MATH_FUNC constexpr size_t        size() const;
    MATH_FUNC constexpr size_t        max_size() const;
#else
    MATH_FUNC bool                    empty() const;
    MATH_FUNC size_t                  size() const;
    MATH_FUNC size_t                  max_size() const;
#endif

    MATH_FUNC void                    fill(T const& value);
    MATH_FUNC void                    swap(array& rhs);

    // Public, to allow for aggregate initialization!
    T data_[N];
};

} // visionaray

#include "detail/array.inl"

#endif // VSNRAY_MATH_ARRAY_H
