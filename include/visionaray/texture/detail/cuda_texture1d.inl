// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstddef>
#include <utility>

#include "../../cuda/array.h"
#include "../../cuda/texture_object.h"
#include "../../array.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// CUDA texture1d
//

template <typename T>
class cuda_texture<T, 1>
{
public:

    using value_type  = T;
    using ref_type    = cuda_texture_ref<T, 1>;
    enum { dimensions = 1 };

private:

    using cuda_type   = typename cuda::map_texel_type<T, cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)>::cuda_type;

public:

    cuda_texture() = default;

    // Only allocate texture
    explicit cuda_texture(size_t w) 
        : width_(w)
    {
        if (width_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = create_channel_desc_with_validation();

        if ( array_.allocate(desc, width_) != cudaSuccess )
        {
            return;
        }
    }

    // Construct from pointer to host data
    template <typename U>
    cuda_texture(
            U const*                           data,
            size_t                             w,
            array<tex_address_mode, 1> const&  address_mode,
            tex_filter_mode const&             filter_mode,
            tex_color_space const&             color_space = RGB,
            bool                               normalized_coords = true
            )
        : width_(w)
        , address_mode_(address_mode)
        , filter_mode_(filter_mode)
        , color_space_(color_space)
        , normalized_coords_(normalized_coords)
    {
        if (width_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = create_channel_desc_with_validation();

        if ( array_.allocate(desc, width_) != cudaSuccess )
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
            tex_address_mode    address_mode,
            tex_filter_mode     filter_mode,
            tex_color_space     color_space = RGB,
            bool                normalized_coords = true
            )
        : width_(w)
        , address_mode_{{ address_mode }}
        , filter_mode_(filter_mode)
        , color_space_(color_space)
        , normalized_coords_(normalized_coords)
    {
        if (width_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = create_channel_desc_with_validation();

        if ( array_.allocate(desc, width_) != cudaSuccess )
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
    explicit cuda_texture(texture<U, 1> const& host_tex)
        : width_(host_tex.width())
        , address_mode_(host_tex.get_address_mode())
        , filter_mode_(host_tex.get_filter_mode())
        , color_space_(host_tex.get_color_space())
        , normalized_coords_(host_tex.get_normalized_coords())
    {
        if (width_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = create_channel_desc_with_validation();

        if ( array_.allocate(desc, width_) != cudaSuccess )
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
    explicit cuda_texture(texture_ref<U, 1> const& host_tex)
        : width_(host_tex.width())
        , address_mode_(host_tex.get_address_mode())
        , filter_mode_(host_tex.get_filter_mode())
        , color_space_(host_tex.get_color_space())
        , normalized_coords_(host_tex.get_normalized_coords())
    {
        if (width_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = create_channel_desc_with_validation();

        if ( array_.allocate(desc, width_) != cudaSuccess )
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
        : array_(std::move(rhs.array_))
        , texture_obj_(std::move(rhs.texture_obj_))
        , width_(rhs.width_)
    {
    }

    cuda_texture& operator=(cuda_texture&& rhs)
    {
        array_ = std::move(rhs.array_);
        texture_obj_ = std::move(rhs.texture_obj_);
        width_ = rhs.width_;

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

    void resize(size_t width)
    {
        width_ = width;

        if (width_ == 0)
        {
            return;
        }

        cudaChannelFormatDesc desc = create_channel_desc_with_validation();

        if ( array_.allocate(desc, width_) != cudaSuccess )
        {
            return;
        }
    }

    // reset from host texture:
    template <typename U>
    void reset(texture<U, 1> const& host_tex)
    {
        if (!(*this) ||
            width_ != host_tex.width() ||
            address_mode_ !=  host_tex.get_address_mode() ||
            filter_mode_ !=  host_tex.get_filter_mode() ||
            color_space_ !=  host_tex.get_color_space() ||
            normalized_coords_ !=  host_tex.get_normalized_coords())
        {
            width_ = host_tex.width();
            address_mode_ =  host_tex.get_address_mode();
            filter_mode_ =  host_tex.get_filter_mode();
            color_space_ =  host_tex.get_color_space();
            normalized_coords_ =  host_tex.get_normalized_coords();

            if (width_ == 0)
            {
                return;
            }

            cudaChannelFormatDesc desc = create_channel_desc_with_validation();

            if ( array_.allocate(desc, width_) != cudaSuccess )
            {
                return;
            }
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
        assert( index < 1 );
        address_mode_[index] = mode;

        init_texture_object();
    }

    void set_address_mode(tex_address_mode mode)
    {
        for (size_t d = 0; d < 1; ++d)
        {
            address_mode_[d] = mode;
        }

        init_texture_object();
    }

    void set_address_mode(array<tex_address_mode, 1> const& mode)
    {
        address_mode_ = mode;

        init_texture_object();
    }

    tex_address_mode get_address_mode(size_t index) const
    {
        assert(index < 1);

        return address_mode_[index];
    }

    array<tex_address_mode, 1> const& get_address_mode() const
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

    bool get_normalized_coord() const
    {
        return normalized_coords_;
    }

    operator bool() const
    {
        return array_.get() != nullptr;
    }

private:

    cuda::array                array_;

    cuda::texture_object       texture_obj_;

    size_t                     width_;

    array<tex_address_mode, 1> address_mode_;
    tex_filter_mode            filter_mode_;
    tex_color_space            color_space_ = RGB;
    bool                       normalized_coords_ = true;


    cudaError_t upload_data(T const* data)
    {
        // Cast from host type to device type
        return array_.upload( reinterpret_cast<cuda_type const*>(data), width_ * sizeof(cuda_type) );
    }

    template <typename U>
    cudaError_t upload_data(U const* data)
    {
        // First promote to host type
        aligned_vector<T> dst( width_ );

        for (size_t i = 0; i < width_; ++i)
        {
            dst[i] = T( data[i] );
        }

        return upload_data( dst.data() );
    }

    cudaChannelFormatDesc create_channel_desc_with_validation()
    {
        // Validate cuda_type
        static_assert(sizeof(cuda_type) > 0, "Invalid cuda_type: type must have non-zero size");
        
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<cuda_type>();
        
        // Check if descriptor is valid (non-zero)
        if (desc.x == 0 && desc.y == 0 && desc.z == 0 && desc.w == 0)
        {
            fprintf(stderr, "%s\n", "WARNING: cudaCreateChannelDesc returned zero descriptor!");
            fprintf(stderr, "%s\n", "This usually means cuda_type is not a standard CUDA type.");

            // For 16-byte types, try assuming it's a 4-component float type
            if (sizeof(cuda_type) == 16)
            {
                fprintf(stderr, "%s\n", "Attempting fallback: creating descriptor for 4x32-bit floats");
                desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            }
        }
        
        return desc;
    }

    cudaError_t init_texture_object()
    {
        cudaResourceDesc resource_desc;
        memset(&resource_desc, 0, sizeof(resource_desc));
        resource_desc.resType                   = cudaResourceTypeArray;
        resource_desc.res.array.array           = array_.get();

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0]             = detail::map_address_mode( address_mode_[0] );
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
// CUDA texture1d reference
//

template <typename T>
class cuda_texture_ref<T, 1>
{
public:

    using value_type  = T;
    using cuda_type   = typename cuda::map_texel_type<T, cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)>::cuda_type;
    enum { dimensions = 1 };

public:

    // Default ctor, dtor and assignment

    cuda_texture_ref()                                         = default;
    cuda_texture_ref(cuda_texture_ref<T, 1> const&)            = default;
    cuda_texture_ref(cuda_texture_ref<T, 1>&&)                 = default;
    cuda_texture_ref& operator=(cuda_texture_ref<T, 1> const&) = default;
    cuda_texture_ref& operator=(cuda_texture_ref<T, 1>&&)      = default;
   ~cuda_texture_ref()                                         = default;


    // Construct / assign from cuda_texture

    VSNRAY_CPU_FUNC cuda_texture_ref(cuda_texture<T, 1> const& rhs)
        : texture_obj_(rhs.texture_object())
        , width_(rhs.width())
    {
    }

    VSNRAY_CPU_FUNC cuda_texture_ref& operator=(cuda_texture<T, 1> const& rhs)
    {
        texture_obj_ = rhs.texture_object();
        width_ = rhs.width();
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

    VSNRAY_FUNC operator bool() const
    {
        return texture_obj_ != 0;
    }

private:

    cudaTextureObject_t texture_obj_;

    size_t width_;

};

} // visionaray
