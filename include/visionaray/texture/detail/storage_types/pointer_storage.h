// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H
#define VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H 1

#include <cstddef>
#include <type_traits>

#include "../../../array.h"
#include "../../../math/simd/gather.h"
#include "../../../math/simd/type_traits.h"

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

    VSNRAY_FUNC
    explicit pointer_storage(array<unsigned, Dim> size)
        : data_(nullptr)
        , size_(size)
    {
    }

    VSNRAY_FUNC
    explicit pointer_storage(T const* data, array<unsigned, Dim> size)
        : data_(data)
        , size_(size)
    {
    }

    // For backwards-compatibilit
    VSNRAY_FUNC
    explicit pointer_storage(unsigned w)
    {
        size_[0] = w;
    }

    VSNRAY_FUNC
    explicit pointer_storage(unsigned w, unsigned h)
    {
        size_[0] = w;
        size_[1] = h;
    }

    VSNRAY_FUNC
    explicit pointer_storage(unsigned w, unsigned h, unsigned d)
    {
        size_[0] = w;
        size_[1] = h;
        size_[2] = d;
    }

    VSNRAY_FUNC
    array<unsigned, Dim> size() const
    {
        return size_;
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    VSNRAY_FUNC
    U value(U /* */, I const& x) const
    {
        return access(U{}, size_t(x));
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    VSNRAY_FUNC
    U value(U /* */, I const& x, I const& y) const
    {
        return access(U{}, y * size_t(size()[0]) + x);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    VSNRAY_FUNC
    U value(U /* */, I const& x, I const& y, I const& z) const
    {
        return access(U{}, z * size_t(size()[0]) * I(size()[1]) + y * I(size()[0]) + x);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type,
        typename = void
        >
    VSNRAY_FUNC
    U value(U /* */, I const& x) const
    {
        return access(U{}, x);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type,
        typename = void
        >
    VSNRAY_FUNC
    U value(U /* */, I const& x, I const& y) const
    {
        return access(U{}, y * I(size()[0]) + x);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type,
        typename = void
        >
    VSNRAY_FUNC
    U value(U /* */, I const& x, I const& y, I const& z) const
    {
        return access(U{}, z * I(size()[0]) * I(size()[1]) + y * I(size()[0]) + x);
    }

    VSNRAY_FUNC
    void reset(T const* data)
    {
        data_ = data;
    }

    VSNRAY_FUNC
    T const* data() const
    {
        return data_;
    }

    VSNRAY_FUNC
    operator bool() const
    {
        return data_ != nullptr;
    }

protected:

    T const* data_ = nullptr;
    array<unsigned, Dim> size_ {{ 0 }};

    template <typename U>
    VSNRAY_FUNC
    U access(U /* */, size_t index) const
    {
        return U(data_[index]);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type
        >
    VSNRAY_FUNC
    U access(U /* */, I const& index) const
    {
        return U(gather(data_, index));
    }

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H
