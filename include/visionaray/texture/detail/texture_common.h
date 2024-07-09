// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H
#define VSNRAY_TEXTURE_DETAIL_TEXTURE_COMMON_H 1

#include <cassert>
#include <cstddef>

#include <algorithm>

#include "../../array.h"
#include "../../math/detail/math.h"
#include "../../math/vector.h"

#include "storage_types/aligned_storage.h"
#include "storage_types/pointer_storage.h"

namespace visionaray
{

//--------------------------------------------------------------------------------------------------
//
//

enum tex_address_mode
{
    Wrap = 0,
    Mirror,
    Clamp,
    Border
};


enum tex_filter_mode
{
    Nearest = 0,
    Linear,
    BSpline,
    BSplineInterpol,
    CardinalSpline
};

enum tex_color_space
{
    RGB = 0,
    sRGB
};

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
    VSNRAY_FUNC
    explicit texture_base(texture_base<Dim, OtherStorage> const& other)
        : TextureStorage(other.data(), other.size())
        , address_mode_(other.get_address_mode())
        , filter_mode_(other.get_filter_mode())
        , color_space_(other.get_color_space())
        , normalized_coords_(other.get_normalized_coords())
    {
    }

    VSNRAY_FUNC texture_base(texture_base<Dim, TextureStorage> const& other) = default;
    VSNRAY_FUNC texture_base(texture_base<Dim, TextureStorage>&& other) = default;

    VSNRAY_FUNC texture_base& operator=(texture_base<Dim, TextureStorage> const& other) = default;
    VSNRAY_FUNC texture_base& operator=(texture_base<Dim, TextureStorage>&& other) = default;

    // Applies the appropriate texture address mode to
    // boundary texture coordinates
    template <typename CoordType>
    VSNRAY_FUNC
    CoordType remap_texture_coordinate(CoordType coord) const
    {
        using F = typename CoordType::value_type;
        using I = decltype(convert_to_int(F{}));

        CoordType result;

        for (unsigned d = 0; d < Dim; ++d)
        {
            int texsize = static_cast<int>(TextureStorage::size()[d]);
            F N = convert_to_float((int)texsize);

            switch (address_mode_[d])
            {
            case Mirror:
                result[d] = select(
                    (convert_to_int(floor(coord[d])) & I(1)) == 1, // if is odd
                    convert_to_float(texsize - 1) / convert_to_float(texsize) - (coord[d] - floor(coord[d])),
                    coord[d] - floor(coord[d])
                    );
                break;

            case Wrap:
                result[d] = coord[d] - floor(coord[d]);
                break;

            case Clamp:
                // fall-through
            default:
                result[d] = clamp(coord[d], F(0.0), F(1.0) - F(1.0) / N);
                break;
            }
        }

        return result;
    }

    // For compatibility
    VSNRAY_FUNC
    inline unsigned width() const
    {
        return TextureStorage::size()[0];
    }

    VSNRAY_FUNC
    inline unsigned height() const
    {
        return TextureStorage::size()[1];
    }

    VSNRAY_FUNC
    inline unsigned depth() const
    {
        return TextureStorage::size()[2];
    }

    VSNRAY_FUNC
    void set_address_mode(unsigned index, tex_address_mode mode)
    {
        assert(index < Dim);
        address_mode_[index] = mode;
    }

    VSNRAY_FUNC
    void set_address_mode(tex_address_mode mode)
    {
        for (unsigned d = 0; d < Dim; ++d)
        {
            address_mode_[d] = mode;
        }
    }

    VSNRAY_FUNC
    void set_address_mode(array<tex_address_mode, Dim> const& mode)
    {
        address_mode_ = mode;
    }

    VSNRAY_FUNC
    tex_address_mode get_address_mode(unsigned index) const
    {
        assert(index < Dim);
        return address_mode_[index];
    }

    VSNRAY_FUNC
    array<tex_address_mode, Dim> const& get_address_mode() const
    {
        return address_mode_;
    }

    VSNRAY_FUNC
    void set_filter_mode(tex_filter_mode mode)
    {
        filter_mode_ = mode;
    }

    VSNRAY_FUNC
    tex_filter_mode get_filter_mode() const
    {
        return filter_mode_;
    }

    VSNRAY_FUNC
    void set_color_space(tex_color_space cs)
    {
        color_space_ = cs;
    }

    VSNRAY_FUNC
    tex_color_space get_color_space() const
    {
        return color_space_;
    }

    VSNRAY_FUNC
    void set_normalized_coords(bool nc)
    {
        normalized_coords_ = nc;
    }

    VSNRAY_FUNC
    bool get_normalized_coords() const
    {
        return normalized_coords_;
    }

protected:

    array<tex_address_mode, Dim> address_mode_;
    tex_filter_mode              filter_mode_;
    tex_color_space              color_space_ = RGB;
    bool                         normalized_coords_ = true;

};


//-------------------------------------------------------------------------------------------------
// Use this class as a view to linear (user-managed) memory
// Expected memory layout: textures are stored one row after another (2D)
// 3D textures are stored slice by slice, where each slice is a 2D texture
//

template <typename T, unsigned Dim>
struct texture_ref : pointer_storage<T, Dim>
{
    using value_type = T;
    using base_type = pointer_storage<T, Dim>;
    enum { dimensions = Dim };
    using base_type::base_type;

    texture_ref() = default;

    texture_ref(unsigned size[Dim])
        : base_type(size)
    {
    }

    // Conversion from another texture type;
    // This is e.g. used for ref()'ing
    template <typename OtherStorage>
    VSNRAY_FUNC
    explicit texture_ref(texture_base<Dim, OtherStorage> const& other)
        : base_type(other.data(), other.size())
        , address_mode_(other.get_address_mode())
        , filter_mode_(other.get_filter_mode())
        , color_space_(other.get_color_space())
        , normalized_coords_(other.get_normalized_coords())
    {
    }

    VSNRAY_FUNC texture_ref(texture_ref<T, Dim> const& other) = default;
    VSNRAY_FUNC texture_ref(texture_ref<T, Dim>&& other) = default;

    VSNRAY_FUNC texture_ref& operator=(texture_ref<T, Dim> const& other) = default;
    VSNRAY_FUNC texture_ref& operator=(texture_ref<T, Dim>&& other) = default;

    // Applies the appropriate texture address mode to
    // boundary texture coordinates
    template <typename CoordType>
    VSNRAY_FUNC
    CoordType remap_texture_coordinate(CoordType coord) const
    {
        using F = typename CoordType::value_type;
        using I = decltype(convert_to_int(F{}));

        CoordType result;

        for (unsigned d = 0; d < Dim; ++d)
        {
            int texsize = static_cast<int>(base_type::size()[d]);
            F N = convert_to_float((int)texsize);

            switch (address_mode_[d])
            {
            case Mirror:
                result[d] = select(
                    (convert_to_int(floor(coord[d])) & I(1)) == 1, // if is odd
                    convert_to_float(texsize - 1) / convert_to_float(texsize) - (coord[d] - floor(coord[d])),
                    coord[d] - floor(coord[d])
                    );
                break;

            case Wrap:
                result[d] = coord[d] - floor(coord[d]);
                break;

            case Clamp:
                // fall-through
            default:
                result[d] = clamp(coord[d], F(0.0), F(1.0) - F(1.0) / N);
                break;
            }
        }

        return result;
    }

    // For compatibility
    VSNRAY_FUNC
    inline unsigned width() const
    {
        return base_type::size()[0];
    }

    VSNRAY_FUNC
    inline unsigned height() const
    {
        return base_type::size()[1];
    }

    VSNRAY_FUNC
    inline unsigned depth() const
    {
        return base_type::size()[2];
    }

    VSNRAY_FUNC
    void set_address_mode(unsigned index, tex_address_mode mode)
    {
        assert(index < Dim);
        address_mode_[index] = mode;
    }

    VSNRAY_FUNC
    void set_address_mode(tex_address_mode mode)
    {
        for (unsigned d = 0; d < Dim; ++d)
        {
            address_mode_[d] = mode;
        }
    }

    VSNRAY_FUNC
    void set_address_mode(array<tex_address_mode, Dim> const& mode)
    {
        address_mode_ = mode;
    }

    VSNRAY_FUNC
    tex_address_mode get_address_mode(unsigned index) const
    {
        assert(index < Dim);
        return address_mode_[index];
    }

    VSNRAY_FUNC
    array<tex_address_mode, Dim> const& get_address_mode() const
    {
        return address_mode_;
    }

    VSNRAY_FUNC
    void set_filter_mode(tex_filter_mode mode)
    {
        filter_mode_ = mode;
    }

    VSNRAY_FUNC
    tex_filter_mode get_filter_mode() const
    {
        return filter_mode_;
    }

    VSNRAY_FUNC
    void set_color_space(tex_color_space cs)
    {
        color_space_ = cs;
    }

    VSNRAY_FUNC
    tex_color_space get_color_space() const
    {
        return color_space_;
    }

    VSNRAY_FUNC
    void set_normalized_coords(bool nc)
    {
        normalized_coords_ = nc;
    }

    VSNRAY_FUNC
    bool get_normalized_coords() const
    {
        return normalized_coords_;
    }
private:

    array<tex_address_mode, Dim> address_mode_;
    tex_filter_mode              filter_mode_;
    tex_color_space              color_space_;
    bool                         normalized_coords_;
};

//-------------------------------------------------------------------------------------------------
// Texture storage class. Use texture_ref<T, Dim> as a view to the data
//

template <typename T, unsigned Dim>
struct texture : texture_base<Dim, aligned_storage<T, Dim, 16>>
{
    using value_type = T;
    using base_type = texture_base<Dim, aligned_storage<T, Dim, 16>>;
    enum { dimensions = Dim };
    using base_type::base_type;
    using ref_type = texture_ref<T, Dim>;
};

// Specialization, uses tiling (TODO..)


// Specialization, uses bricking
// template <typename T>
// struct texture<T, 3> : texture_base<3, aligned_storage<T, 3, 16>>
// {
//     using value_type = T;
//     using base_type = texture_base<3, aligned_storage<T, 3, 16>>;
//     enum { dimensions = 3 };
//     using base_type::base_type;
//     using ref_type = texture_ref<T, 3>;
// };


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
