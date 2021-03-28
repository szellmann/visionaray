// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H
#define VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H 1

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <array>

#include <visionaray/math/norm.h>
#include <visionaray/math/vector.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/pixel_format.h>
#include <visionaray/swizzle.h>

#include "../forward.h"

namespace visionaray
{

template <size_t Dim, typename TextureStorage>
class texture_base : public TextureStorage
{
public:
    using TextureStorage::TextureStorage;

public:

    texture_base() = default;

    texture_base(unsigned size[Dim])
        : TextureStorage(size)
    {
    }

    // Conversion from one texture type to another; no
    // guarantees that this might work!
    // This is e.g. used for ref()'ing
    template <typename OtherStorage>
    explicit texture_base(texture_base<Dim, OtherStorage> const& other)
        : TextureStorage(other.data(), other.size())
        , address_mode_(other.get_address_mode())
        , filter_mode_(other.get_filter_mode())
        , color_space_(other.get_color_space())
        , normalized_coords_(other.get_normalized_coords())
    {
    }

    texture_base(texture_base<Dim, TextureStorage> const& other) = default;
    texture_base(texture_base<Dim, TextureStorage>&& other) = default;

    texture_base& operator=(texture_base<Dim, TextureStorage> const& other) = default;
    texture_base& operator=(texture_base<Dim, TextureStorage>&& other) = default;

    // For compatibility
    inline unsigned width() const
    {
        return TextureStorage::size()[0];
    }

    inline unsigned height() const
    {
        return TextureStorage::size()[1];
    }

    inline unsigned depth() const
    {
        return TextureStorage::size()[2];
    }

    void set_address_mode(size_t index, tex_address_mode mode)
    {
        assert(index < Dim);
        address_mode_[index] = mode;
    }

    void set_address_mode(tex_address_mode mode)
    {
        for (size_t d = 0; d < Dim; ++d)
        {
            address_mode_[d] = mode;
        }
    }

    void set_address_mode(std::array<tex_address_mode, Dim> const& mode)
    {
        address_mode_ = mode;
    }

    tex_address_mode get_address_mode(size_t index) const
    {
        assert(index < Dim);
        return address_mode_[index];
    }

    std::array<tex_address_mode, Dim> const& get_address_mode() const
    {
        return address_mode_;
    }

    void set_filter_mode(tex_filter_mode mode)
    {
        filter_mode_ = mode;
    }

    tex_filter_mode get_filter_mode() const
    {
        return filter_mode_;
    }

    void set_color_space(tex_color_space cs)
    {
        color_space_ = cs;
    }

    tex_color_space get_color_space() const
    {
        return color_space_;
    }

    void set_normalized_coords(bool nc)
    {
        normalized_coords_ = nc;
    }

    bool get_normalized_coords() const
    {
        return normalized_coords_;
    }

protected:

    std::array<tex_address_mode, Dim> address_mode_;
    tex_filter_mode                   filter_mode_;
    tex_color_space                   color_space_ = RGB;
    bool                              normalized_coords_ = true;

};


//-------------------------------------------------------------------------------------------------
// Simple linear storage type. Data is aligned to allow for SIMD access
//

template <typename T, size_t Dim, size_t A = 16>
class aligned_storage
{
public:

    using value_type = T;

public:

    aligned_storage() = default;

    explicit aligned_storage(std::array<unsigned, Dim> size)
        : data_(linear_size(size))
        , size_(size)
    {
    }

    explicit aligned_storage(unsigned w)
    {
        realloc(w);
    }

    explicit aligned_storage(unsigned w, unsigned h)
    {
        realloc(w, h);
    }

    explicit aligned_storage(unsigned w, unsigned h, unsigned d)
    {
        realloc(w, h, d);
    }

    std::array<unsigned, Dim> size() const
    {
        return size_;
    }

    void realloc(unsigned w)
    {
        size_[0] = w;
        data_.resize(linear_size(size_));
    }

    void realloc(unsigned w, unsigned h)
    {
        size_[0] = w;
        size_[1] = h;
        data_.resize(linear_size(size_));
    }

    void realloc(unsigned w, unsigned h, unsigned d)
    {
        size_[0] = w;
        size_[1] = h;
        size_[2] = d;
        data_.resize(linear_size(size_));
    }

    void reset(T const* data)
    {
        std::copy(data, data + data_.size(), data_.begin());
    }

    void reset(
            T const* data,
            pixel_format format,
            pixel_format internal_format
            )
    {
        if (format != internal_format)
        {
            // Swizzle in-place
            aligned_vector<T> tmp(data, data + data_.size());
            swizzle(tmp.data(), internal_format, format, tmp.size());
            reset(tmp.data());
        }
        else
        {
            // Simple copy
            reset(data);
        }
    }

    template <typename U>
    void reset(
            U const* data,
            pixel_format format,
            pixel_format internal_format
            )
    {
        // Copy to temporary array, then swizzle
        aligned_vector<T> dst(data_.size());
        swizzle(dst.data(), internal_format, data, format, dst.size());
        reset(dst.data());
    }

    template <typename U>
    void reset(
            U const* data,
            pixel_format format,
            pixel_format internal_format,
            swizzle_hint hint
            )
    {
        // Copy with temporary array, hint about how to handle alpha
        aligned_vector<T> dst(data_.size());
        swizzle(dst.data(), internal_format, data, format, dst.size(), hint);
        reset(dst.data());
    }

    value_type const* data() const
    {
        return data_.data();
    }

    operator bool() const
    {
        return !data_.empty();
    }

protected:

    inline size_t linear_size(std::array<unsigned, 1> size)
    {
        return size_t(size[0]);
    }

    inline size_t linear_size(std::array<unsigned, 2> size)
    {
        return size[0] * size_t(size[1]);
    }

    inline size_t linear_size(std::array<unsigned, 3> size)
    {
        return size[0] * size[1] * size_t(size[2]);
    }

    aligned_vector<T, A> data_;
    std::array<unsigned, Dim> size_;

};


//-------------------------------------------------------------------------------------------------
// Storage that is managed by the user; we only store a pointer and in addition know
// what size the pointee is supposed to have
//

template <typename T, size_t Dim>
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


template <typename T, size_t Dim>
struct texture_ref : texture_base<Dim, pointer_storage<T, Dim>>
{
    using base_type = texture_base<Dim, pointer_storage<T, Dim>>;
    enum { dimensions = Dim };
    using base_type::base_type;
};

template <typename T, size_t Dim>
struct texture : texture_base<Dim, aligned_storage<T, Dim, 16>>
{
    using base_type = texture_base<Dim, aligned_storage<T, Dim, 16>>;
    enum { dimensions = Dim };
    using base_type::base_type;
    using ref_type = texture_ref<T, Dim>;
};


template <typename T>
VSNRAY_FUNC
inline T apply_color_conversion(T const& t, tex_color_space const& color_space)
{
    VSNRAY_UNUSED(color_space);

    return t;
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> apply_color_conversion(vector<3, T> const& t, tex_color_space const& color_space)
{
    if (color_space == sRGB)
    {
        return vector<3, T>(
                pow(t.x, T(2.2)),
                pow(t.y, T(2.2)),
                pow(t.z, T(2.2))
                );
    }
    else
    {
        return t;
    }
}

template <typename T>
VSNRAY_FUNC
inline vector<4, T> apply_color_conversion(vector<4, T> const& t, tex_color_space const& color_space)
{
    if (color_space == sRGB)
    {
        return vector<4, T>(
                pow(t.x, T(2.2)),
                pow(t.y, T(2.2)),
                pow(t.z, T(2.2)),
                t.w
                );
    }
    else
    {
        return t;
    }
}

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H
