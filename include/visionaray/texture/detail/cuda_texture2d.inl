// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <visionaray/cuda/pitch2d.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// CUDA texture2d
//

template <typename T>
class cuda_texture<T, 2>
{
public:

    using value_type  = T;
    using ref_type    = cuda_texture_ref<T, 2>;
    enum { dimensions = 2 };

private:

    using cuda_type   = typename cuda::map_texel_type<T, cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)>::cuda_type;

public:

    cuda_texture() = default;

    // Only allocate texture
    cuda_texture(size_t w, size_t h)
        : width_(w)
        , height_(h)
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        if ( pitch_.allocate(width_, height_) != cudaSuccess )
        {
            return;
        }
    }

    // Construct from pointer to host data
    template <typename U>
    cuda_texture(
            U const*                                data,
            size_t                                  w,
            size_t                                  h,
            std::array<tex_address_mode, 2> const&  address_mode,
            tex_filter_mode const&                  filter_mode,
            tex_color_space const&                  color_space = RGB,
            bool                                    normalized_coords = true
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

        if ( pitch_.allocate(width_, height_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(data) != cudaSuccess )
        {
            return;
        }

        if ( init_texture_object() != cudaSuccess )
        {
            return;
        }
    }

    // Construct from pointer to host data, same address mode for all dimensions
    template <typename U>
    cuda_texture(
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

        if ( pitch_.allocate(width_, height_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(data) != cudaSuccess )
        {
            return;
        }

        if ( init_texture_object() != cudaSuccess )
        {
            return;
        }
    }

    // Construct from host texture
    template <typename U>
    explicit cuda_texture(texture<U, 2> const& host_tex)
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

        if ( pitch_.allocate(width_, height_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != cudaSuccess )
        {
            return;
        }

        if ( init_texture_object() != cudaSuccess )
        {
            return;
        }
    }

    // Construct from host texture ref (TODO: combine with previous)
    template <typename U>
    explicit cuda_texture(texture_ref<U, 2> const& host_tex)
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

        if ( pitch_.allocate(width_, height_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != cudaSuccess )
        {
            return;
        }

        if ( init_texture_object() != cudaSuccess )
        {
            return;
        }
    }

#if !VSNRAY_CXX_MSVC
    cuda_texture(cuda_texture&&) = default;
    cuda_texture& operator=(cuda_texture&&) = default;
#else
    cuda_texture(cuda_texture&& rhs)
        : pitch_(std::move(rhs.pitch_))
        , texture_obj_(std::move(rhs.texture_obj_))
        , width_(rhs.width_)
        , height_(rhs.height_)
    {
    }

    cuda_texture& operator=(cuda_texture&& rhs)
    {
        pitch_ = std::move(rhs.pitch_);
        texture_obj_ = std::move(rhs.texture_obj_);
        width_ = rhs.width_;
        height_ = rhs.height_;

        return *this;
    }
#endif

    // NOT copyable
    cuda_texture(cuda_texture const& rhs) = delete;
    cuda_texture& operator=(cuda_texture const& rhs) = delete;


    cudaTextureObject_t texture_object() const
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

        if ( pitch_.allocate(width_, height_) != cudaSuccess )
        {
            return;
        }
    }

    template <typename U>
    void reset(U const* data)
    {
        if ( upload_data(data) != cudaSuccess )
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

    void set_address_mode(std::array<tex_address_mode, 2> const& mode)
    {
        address_mode_ = mode;

        init_texture_object();
    }

    void set_filter_mode(tex_filter_mode filter_mode)
    {
        filter_mode_ = filter_mode;

        init_texture_object();
    }

    void set_color_space(tex_color_space color_space)
    {
        color_space_ = color_space;

        init_texture_object();
    }

    void set_normalized_coords(bool nc)
    {
        normalized_coords_ = nc;

        init_texture_object();
    }

    operator bool() const
    {
        return pitch_.get() != nullptr;
    }

private:

    cuda::pitch2d<cuda_type>        pitch_;

    cuda::texture_object            texture_obj_;

    size_t                          width_;
    size_t                          height_;

    std::array<tex_address_mode, 2> address_mode_;
    tex_filter_mode                 filter_mode_;
    tex_color_space                 color_space_ = RGB;
    bool                            normalized_coords_ = true;


    cudaError_t upload_data(T const* data)
    {
        // Cast from host type to device type
        return pitch_.upload( reinterpret_cast<cuda_type const*>(data), width_, height_ );
    }

    template <typename U>
    cudaError_t upload_data(U const* data)
    {
        // First promote to host type
        aligned_vector<T> dst( width_ * height_ );

        for (size_t i = 0; i < width_ * height_; ++i)
        {
            dst[i] = T( data[i] );
        }

        return upload_data( dst.data() );
    }

    cudaError_t init_texture_object()
    {
        auto desc = cudaCreateChannelDesc<cuda_type>();

        cudaResourceDesc resource_desc;
        memset(&resource_desc, 0, sizeof(resource_desc));
        resource_desc.resType                   = cudaResourceTypePitch2D;
        resource_desc.res.pitch2D.devPtr        = pitch_.get();
        resource_desc.res.pitch2D.pitchInBytes  = pitch_.get_pitch_in_bytes();
        resource_desc.res.pitch2D.width         = width_;
        resource_desc.res.pitch2D.height        = height_;
        resource_desc.res.pitch2D.desc          = desc;

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0]             = detail::map_address_mode( address_mode_[0] );
        texture_desc.addressMode[1]             = detail::map_address_mode( address_mode_[1] );
        texture_desc.filterMode                 = detail::map_filter_mode( filter_mode_ );
        texture_desc.readMode                   = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value);
        texture_desc.sRGB                       = color_space_ == sRGB;
        texture_desc.normalizedCoords           = normalized_coords_;

        cudaTextureObject_t obj = 0;
        cudaError_t err = cudaCreateTextureObject( &obj, &resource_desc, &texture_desc, 0 );

        if (err == cudaSuccess)
        {
            texture_obj_.reset(obj);
        }

        return err;
    }
};


//-------------------------------------------------------------------------------------------------
// CUDA texture2d reference
//

template <typename T>
class cuda_texture_ref<T, 2>
{
public:

    using value_type  = T;
    using cuda_type   = typename cuda::map_texel_type<T, cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)>::cuda_type;
    enum { dimensions = 2 };

public:

    // Default ctor, dtor and assignment

    cuda_texture_ref()                                         = default;
    cuda_texture_ref(cuda_texture_ref<T, 2> const&)            = default;
    cuda_texture_ref(cuda_texture_ref<T, 2>&&)                 = default;
    cuda_texture_ref& operator=(cuda_texture_ref<T, 2> const&) = default;
    cuda_texture_ref& operator=(cuda_texture_ref<T, 2>&&)      = default;
   ~cuda_texture_ref()                                         = default;


    // Construct assign from cuda_texture

    VSNRAY_CPU_FUNC cuda_texture_ref(cuda_texture<T, 2> const& ref)
        : texture_obj_(ref.texture_object())
        , width_(ref.width())
        , height_(ref.height())
    {
    }


    VSNRAY_CPU_FUNC cuda_texture_ref& operator=(cuda_texture<T, 2> const& rhs)
    {
        texture_obj_ = rhs.texture_object();
        width_ = rhs.width();
        height_ = rhs.height();
        return *this;
    }


    VSNRAY_FUNC cudaTextureObject_t texture_object() const
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

    cudaTextureObject_t texture_obj_;

    size_t width_;
    size_t height_;

};

} // visionaray
