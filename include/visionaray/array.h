// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ARRAY_H
#define VSNRAY_ARRAY_H 1

#include <cstddef>

#include "detail/macros.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// array
//
// Implements the functionality provided by std::array, but with all access functions
// declared as host/device for compatibility with NVIDIA nvcc
//

template <typename T, size_t N>
class array
{
public:

    using value_type        = T;
    using size_type         = size_t;
    using difference_type   = ptrdiff_t;
    using reference         = value_type&;
    using const_reference   = value_type const&;
    using pointer           = T*;
    using const_pointer     = T const*;
    using iterator          = T*;
    using const_iterator    = T const*;
    // TODO: reverse iterators

public:

    VSNRAY_FUNC reference           at(size_type pos);
    VSNRAY_FUNC const_reference     at(size_type pos) const;

    VSNRAY_FUNC reference           operator[](size_type pos);
    VSNRAY_FUNC const_reference     operator[](size_type pos) const;

    VSNRAY_FUNC reference           front();
    VSNRAY_FUNC const_reference     front() const;

    VSNRAY_FUNC reference           back();
    VSNRAY_FUNC const_reference     back() const;

    VSNRAY_FUNC T*                  data();
    VSNRAY_FUNC T const*            data() const;


    VSNRAY_FUNC iterator            begin();
    VSNRAY_FUNC const_iterator      begin() const;
    VSNRAY_FUNC const_iterator      cbegin() const;

    VSNRAY_FUNC iterator            end();
    VSNRAY_FUNC const_iterator      end() const;
    VSNRAY_FUNC const_iterator      cend() const;

    // TODO: reverse iterators


    VSNRAY_FUNC constexpr bool      empty() const;
    VSNRAY_FUNC constexpr size_t    size() const;

    VSNRAY_FUNC void                fill(T const& value);
    VSNRAY_FUNC void                swap(array& rhs);

private:

    T data_[N];

};

} // visionaray

#include "detail/array.inl"

#endif
