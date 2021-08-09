// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H
#define VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H 1

#include <array>
#include <type_traits>

#include <visionaray/math/simd/gather.h>
#include <visionaray/math/simd/type_traits.h>

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

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    U value(U /* */, I const& x) const
    {
        return access(U{}, size_t(x));
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    U value(U /* */, I const& x, I const& y) const
    {
        return access(U{}, y * I(size()[0]) + size_t(x));
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    U value(U /* */, I const& x, I const& y, I const& z) const
    {
        return access(U{}, z * I(size()[0]) * I(size()[1]) + y * I(size()[0]) + size_t(x));
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type,
        typename = void
        >
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
    U value(U /* */, I const& x, I const& y, I const& z) const
    {
        return access(U{}, z * I(size()[0]) * I(size()[1]) + y * I(size()[0]) + x);
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

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<!simd::is_simd_vector<I>::value>::type
        >
    U access(U /* */, I const& index) const
    {
        return U(data_[index]);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type,
        typename = void
        >
    U access(U /* */, I const& index) const
    {
        return U(gather(data_, index));
    }

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_POINTER_STORAGE_H
