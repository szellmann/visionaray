// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H
#define VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H 1

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <array>

#include <visionaray/math/vector.h>

#include "../forward.h"
#include "storage_types/aligned_storage.h"
#include "storage_types/pointer_storage.h"

namespace visionaray
{

template <unsigned Dim, typename TextureStorage>
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

    void set_address_mode(unsigned index, tex_address_mode mode)
    {
        assert(index < Dim);
        address_mode_[index] = mode;
    }

    void set_address_mode(tex_address_mode mode)
    {
        for (unsigned d = 0; d < Dim; ++d)
        {
            address_mode_[d] = mode;
        }
    }

    void set_address_mode(std::array<tex_address_mode, Dim> const& mode)
    {
        address_mode_ = mode;
    }

    tex_address_mode get_address_mode(unsigned index) const
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


template <typename T, unsigned Dim>
struct texture_ref : texture_base<Dim, pointer_storage<T, Dim>>
{
    using base_type = texture_base<Dim, pointer_storage<T, Dim>>;
    enum { dimensions = Dim };
    using base_type::base_type;
};

template <typename T, unsigned Dim>
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
