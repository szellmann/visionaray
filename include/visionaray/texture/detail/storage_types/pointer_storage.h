// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H
#define VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H 1

#include <array>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Storage that is managed by the user; we only store a pointer and in addition know
// what size the pointee is supposed to have
//

template <typename T, unsigned Dim>
class pointer_storage
{
public:

    using value_type = T;
    enum { dimensions = Dim };

public:

    pointer_storage() = default;

    explicit pointer_storage(std::array<unsigned, Dim> size)
        : data_(nullptr)
        , size_(size)
    {
    }

    explicit pointer_storage(T const* data, std::array<unsigned, Dim> size)
        : data_(data)
        , size_(size)
    {
    }

    // For backwards-compatibility
    explicit pointer_storage(unsigned w)
    {
        size_[0] = w;
    }

    explicit pointer_storage(unsigned w, unsigned h)
    {
        size_[0] = w;
        size_[1] = h;
    }

    explicit pointer_storage(unsigned w, unsigned h, unsigned d)
    {
        size_[0] = w;
        size_[1] = h;
        size_[2] = d;
    }

    std::array<unsigned, Dim> size() const
    {
        return size_;
    }

    void reset(T const* data)
    {
        data_ = data;
    }

    T const* data() const
    {
        return data_;
    }

    operator bool() const
    {
        return data_ != nullptr;
    }

protected:

    T const* data_ = nullptr;
    std::array<unsigned, Dim> size_ {{ 0 }};

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H
