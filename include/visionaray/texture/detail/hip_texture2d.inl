// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstddef>
#include <utility>

#include "../../hip/pitch2d.h"
#include "../../hip/texture_object.h"
#include "../../array.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// HIP texture2d
//

template <typename T>
class hip_texture<T, 2>
{
public:

    using value_type  = T;
    using ref_type    = hip_texture_ref<T, 2>;
    enum { dimensions = 2 };

private:

    using hip_type   = typename hip::map_texel_type<T, hipTextureReadMode(detail::tex_read_mode_from_type<T>::value)>::hip_type;

public:

    hip_texture() = default;

    // Only allocate texture
    hip_texture(size_t w, size_t h)
        : width_(w)
        , height_(h)
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != hipSuccess )
        {
            return;
        }
    }

    // Construct from pointer to host data
    template <typename U>
    hip_texture(
            U const*                          data,
            size_t                            w,
            size_t                            h,
            array<tex_address_mode, 2> const& address_mode,
            tex_filter_mode const&            filter_mode,
            tex_color_space const&            color_space = RGB,
            bool                              normalized_coords = true
            )
        : width_(w)
        , height_(h)
        , address_mode_(address_mode)
        , filter_mode_(filter_mode)
        , color_space_(color_space)
        , normalized_coords_(normalized_coords)
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != hipSuccess )
        {
            return;
        }

        if ( upload_data(data) != hipSuccess )
        {
            return;
        }

        if ( init_texture_object() != hipSuccess )
        {
            return;
        }
    }

    // Construct from pointer to host data, same address mode for all dimensions
    template <typename U>
    hip_texture(
            U const*            data,
            size_t              w,
            size_t              h,
            tex_address_mode    address_mode,
            tex_filter_mode     filter_mode,
            tex_color_space     color_space = sRGB,
            bool                normalized_coords = true
            )
        : width_(w)
        , height_(h)
        , address_mode_{{ address_mode, address_mode }}
        , filter_mode_(filter_mode)
        , color_space_(color_space)
        , normalized_coords_(normalized_coords)
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != hipSuccess )
        {
            return;
        }

        if ( upload_data(data) != hipSuccess )
        {
            return;
        }

        if ( init_texture_object() != hipSuccess )
        {
            return;
        }
    }

    // Construct from host texture
    template <typename U>
    explicit hip_texture(texture<U, 2> const& host_tex)
        : width_(host_tex.width())
        , height_(host_tex.height())
        , address_mode_(host_tex.get_address_mode())
        , filter_mode_(host_tex.get_filter_mode())
        , color_space_(host_tex.get_color_space())
        , normalized_coords_(host_tex.get_normalized_coords())
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != hipSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != hipSuccess )
        {
            return;
        }

        if ( init_texture_object() != hipSuccess )
        {
            return;
        }
    }

    // Construct from host texture ref (TODO: combine with previous)
    template <typename U>
    explicit hip_texture(texture_ref<U, 2> const& host_tex)
        : width_(host_tex.width())
        , height_(host_tex.height())
        , address_mode_(host_tex.get_address_mode())
        , filter_mode_(host_tex.get_filter_mode())
        , color_space_(host_tex.get_color_space())
        , normalized_coords_(host_tex.get_normalized_coords())
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != hipSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != hipSuccess )
        {
            return;
        }

        if ( init_texture_object() != hipSuccess )
        {
            return;
        }
    }

#if !VSNRAY_CXX_MSVC
    hip_texture(hip_texture&&) = default;
    hip_texture& operator=(hip_texture&&) = default;
#else
    hip_texture(hip_texture&& rhs)
        : pitch_(std::move(rhs.pitch_))
        , texture_obj_(std::move(rhs.texture_obj_))
        , width_(rhs.width_)
        , height_(rhs.height_)
    {
    }

    hip_texture& operator=(hip_texture&& rhs)
    {
        pitch_ = std::move(rhs.pitch_);
        texture_obj_ = std::move(rhs.texture_obj_);
        width_ = rhs.width_;
        height_ = rhs.height_;

        return *this;
    }
#endif

    // NOT copyable
    hip_texture(hip_texture const& rhs) = delete;
    hip_texture& operator=(hip_texture const& rhs) = delete;


    hipTextureObject_t texture_object() const
    {
        return texture_obj_.get();
    }

    size_t width() const
    {
        return width_;
    }

    size_t height() const
    {
        return height_;
    }

    void resize(size_t width, size_t height)
    {
        width_  = width;
        height_ = height;

        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != hipSuccess )
        {
            return;
        }
    }

    template <typename U>
    void reset(U const* data)
    {
        if ( upload_data(data) != hipSuccess )
        {
            return;
        }

        init_texture_object();
    }

    void set_address_mode(size_t index, tex_address_mode mode)
    {
        assert( index < 2 );
        address_mode_[index] = mode;

        init_texture_object();
    }

    void set_address_mode(tex_address_mode mode)
    {
        for (size_t d = 0; d < 2; ++d)
        {
            address_mode_[d] = mode;
        }

        init_texture_object();
    }

    void set_address_mode(array<tex_address_mode, 2> const& mode)
    {
        address_mode_ = mode;

        init_texture_object();
    }

    tex_address_mode get_address_mode(size_t index) const
    {
        assert(index < 2);

        return address_mode_[index];
    }

    array<tex_address_mode, 2> const& get_address_mode() const
    {
        return address_mode_;
    }

    void set_filter_mode(tex_filter_mode filter_mode)
    {
        filter_mode_ = filter_mode;

        init_texture_object();
    }

    tex_filter_mode get_filter_mode() const
    {
        return filter_mode_;
    }

    void set_color_space(tex_color_space color_space)
    {
        color_space_ = color_space;

        init_texture_object();
    }

    tex_color_space get_color_space() const
    {
        return color_space_;
    }

    void set_normalized_coords(bool nc)
    {
        normalized_coords_ = nc;

        init_texture_object();
    }

    bool get_normalized_coords() const
    {
        return normalized_coords_;
    }

    operator bool() const
    {
        return pitch_.get() != nullptr;
    }

private:

    hip::pitch2d<hip_type>     pitch_;

    hip::texture_object        texture_obj_;

    size_t                     width_;
    size_t                     height_;

    array<tex_address_mode, 2> address_mode_;
    tex_filter_mode            filter_mode_;
    tex_color_space            color_space_ = RGB;
    bool                       normalized_coords_ = true;


    hipError_t upload_data(T const* data)
    {
        // Cast from host type to device type
        return pitch_.upload( reinterpret_cast<hip_type const*>(data), width_, height_ );
    }

    template <typename U>
    hipError_t upload_data(U const* data)
    {
        // First promote to host type
        aligned_vector<T> dst( width_ * height_ );

        for (size_t i = 0; i < width_ * height_; ++i)
        {
            dst[i] = T( data[i] );
        }

        return upload_data( dst.data() );
    }

    hipError_t init_texture_object()
    {
        auto desc = hipCreateChannelDesc<hip_type>();

        hipResourceDesc resource_desc;
        memset(&resource_desc, 0, sizeof(resource_desc));
        resource_desc.resType                   = hipResourceTypePitch2D;
        resource_desc.res.pitch2D.devPtr        = pitch_.get();
        resource_desc.res.pitch2D.pitchInBytes  = pitch_.get_pitch_in_bytes();
        resource_desc.res.pitch2D.width         = width_;
        resource_desc.res.pitch2D.height        = height_;
        resource_desc.res.pitch2D.desc          = desc;

        hipTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0]             = detail::map_address_mode( address_mode_[0] );
        texture_desc.addressMode[1]             = detail::map_address_mode( address_mode_[1] );
        texture_desc.filterMode                 = detail::map_filter_mode( filter_mode_ );
        texture_desc.readMode                   = hipTextureReadMode(detail::tex_read_mode_from_type<T>::value);
        texture_desc.sRGB                       = color_space_ == sRGB;
        texture_desc.normalizedCoords           = normalized_coords_;

        hipTextureObject_t obj = 0;
        hipError_t err = hipCreateTextureObject( &obj, &resource_desc, &texture_desc, 0 );

        if (err == hipSuccess)
        {
            texture_obj_.reset(obj);
        }

        return err;
    }
};


//-------------------------------------------------------------------------------------------------
// HIP texture2d reference
//

template <typename T>
class hip_texture_ref<T, 2>
{
public:

    using value_type  = T;
    using hip_type   = typename hip::map_texel_type<T, hipTextureReadMode(detail::tex_read_mode_from_type<T>::value)>::hip_type;
    enum { dimensions = 2 };

public:

    // Default ctor, dtor and assignment

    hip_texture_ref()                                         = default;
    hip_texture_ref(hip_texture_ref<T, 2> const&)            = default;
    hip_texture_ref(hip_texture_ref<T, 2>&&)                 = default;
    hip_texture_ref& operator=(hip_texture_ref<T, 2> const&) = default;
    hip_texture_ref& operator=(hip_texture_ref<T, 2>&&)      = default;
   ~hip_texture_ref()                                         = default;


    // Construct assign from hip_texture

    VSNRAY_CPU_FUNC hip_texture_ref(hip_texture<T, 2> const& ref)
        : texture_obj_(ref.texture_object())
        , width_(ref.width())
        , height_(ref.height())
    {
    }


    VSNRAY_CPU_FUNC hip_texture_ref& operator=(hip_texture<T, 2> const& rhs)
    {
        texture_obj_ = rhs.texture_object();
        width_ = rhs.width();
        height_ = rhs.height();
        return *this;
    }


    VSNRAY_FUNC hipTextureObject_t texture_object() const
    {
        return texture_obj_;
    }

    VSNRAY_FUNC size_t width() const
    {
        return width_;
    }

    VSNRAY_FUNC size_t height() const
    {
        return height_;
    }

    VSNRAY_FUNC operator bool() const
    {
        return texture_obj_ != 0;
    }

private:

    hipTextureObject_t texture_obj_;

    size_t width_;
    size_t height_;

};

} // visionaray
