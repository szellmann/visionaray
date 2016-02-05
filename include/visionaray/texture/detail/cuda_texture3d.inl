// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

#include <visionaray/cuda/array.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// CUDA texture3d
//

template <typename T, tex_read_mode ReadMode>
class cuda_texture<T, ReadMode, 3>
{
public:

    using cuda_type   = typename cuda::map_texel_type<T, ReadMode>::cuda_type;
    using vsnray_type = typename cuda::map_texel_type<T, ReadMode>::vsnray_type;

public:

    cuda_texture() = default;

    // Construct from pointer to host data
    template <typename U>
    cuda_texture(
            U const*                                data,
            size_t                                  w,
            size_t                                  h,
            size_t                                  d,
            std::array<tex_address_mode, 3> const&  address_mode,
            tex_filter_mode const&                  filter_mode
            )
        : width_(w)
        , height_(h)
        , depth_(d)
    {
        if (width_ == 0 || height_ == 0 || depth_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = cudaCreateChannelDesc<cuda_type>();

        if ( array_.allocate3D(desc, width_, height_, depth_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(data) != cudaSuccess )
        {
            return;
        }

        init_texture_object(address_mode, filter_mode);
    }

    // Construct from pointer to host data, same address mode for all dimensions
    template <typename U>
    cuda_texture(
            U const*            data,
            size_t              w,
            size_t              h,
            size_t              d,
            tex_address_mode    address_mode,
            tex_filter_mode     filter_mode
            )
        : width_(w)
        , height_(h)
        , depth_(d)
    {
        if (width_ == 0 || height_ == 0 || depth_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = cudaCreateChannelDesc<cuda_type>();

        if ( array_.allocate3D(desc, width_, height_, depth_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(data) != cudaSuccess )
        {
            return;
        }

        std::array<tex_address_mode, 3> am{{ address_mode, address_mode, address_mode }};
        init_texture_object(am, filter_mode);
    }

    // Construct from host texture
    template <typename U>
    explicit cuda_texture(texture<U, ReadMode, 3> const& host_tex)
        : width_(host_tex.width())
        , height_(host_tex.height())
        , depth_(host_tex.depth())
    {
        if (width_ == 0 || height_ == 0 || depth_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = cudaCreateChannelDesc<cuda_type>();

        if ( array_.allocate3D(desc, width_, height_, depth_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != cudaSuccess )
        {
            return;
        }

        init_texture_object(
                host_tex.get_address_mode(),
                host_tex.get_filter_mode()
                );
    }

    // Construct from host texture ref (TODO: combine with previous)
    template <typename U>
    explicit cuda_texture(texture_ref<U, ReadMode, 3> const& host_tex)
        : width_(host_tex.width())
        , height_(host_tex.height())
        , depth_(host_tex.depth())
    {
        if (width_ == 0 || height_ == 0 || depth_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = cudaCreateChannelDesc<cuda_type>();

        if ( array_.allocate3D(desc, width_, height_, depth_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != cudaSuccess )
        {
            return;
        }

        init_texture_object(
                host_tex.get_address_mode(),
                host_tex.get_filter_mode()
                );
    }

#if !VSNRAY_CXX_MSVC
    cuda_texture(cuda_texture&&) = default;
    cuda_texture& operator=(cuda_texture&&) = default;
#else
    cuda_texture(cuda_texture&& rhs)
        : array_(std::move(rhs.array_))
        , texture_obj_(std::move(rhs.texture_obj_))
        , width_(rhs.width_)
        , height_(rhs.height_)
        , depth_(rhs.depth_)
    {
    }

    cuda_texture& operator=(cuda_texture&& rhs)
    {
        array_ = std::move(rhs.array_);
        texture_obj_ = std::move(rhs.texture_obj_);
        width_ = rhs.width_;
        height_ = rhs.height_;
        depth_ = rhs.depth_;

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

    size_t depth() const
    {
        return depth_;
    }

private:

    cuda::array             array_;

    cuda::texture_object    texture_obj_;

    size_t width_;
    size_t height_;
    size_t depth_;

    cudaError_t upload_data(vsnray_type const* data)
    {
        // Cast from host type to device type
        return array_.upload( reinterpret_cast<cuda_type const*>(data), width_, height_, depth_ );
    }

    template <typename U>
    cudaError_t upload_data(U const* data)
    {
        // First promote to host type
        aligned_vector<vsnray_type> dst( width_ );

        for (size_t i = 0; i < width_ * height_ * depth_; ++i)
        {
            dst[i] = vsnray_type( data[i] );
        }

        return upload_data( dst.data() );
    }

    void init_texture_object(
            std::array<tex_address_mode, 3> const&  address_mode,
            tex_filter_mode                         filter_mode
            )
    {
        cudaResourceDesc resource_desc;
        memset(&resource_desc, 0, sizeof(resource_desc));
        resource_desc.resType                   = cudaResourceTypeArray;
        resource_desc.res.array.array           = array_.get();

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0]             = detail::map_address_mode( address_mode[0] );
        texture_desc.addressMode[1]             = detail::map_address_mode( address_mode[1] );
        texture_desc.addressMode[2]             = detail::map_address_mode( address_mode[2] );
        texture_desc.filterMode                 = detail::map_filter_mode( filter_mode );
        texture_desc.readMode                   = detail::map_read_mode( ReadMode );
        texture_desc.normalizedCoords           = true;

        cudaTextureObject_t obj = 0;
        cudaCreateTextureObject( &obj, &resource_desc, &texture_desc, 0 );
        texture_obj_.reset(obj);
    }

};


//-------------------------------------------------------------------------------------------------
// CUDA texture3d reference
//

template <typename T, tex_read_mode ReadMode>
class cuda_texture_ref<T, ReadMode, 3>
{
public:

    using cuda_type   = typename cuda::map_texel_type<T, ReadMode>::cuda_type;
    using vsnray_type = typename cuda::map_texel_type<T, ReadMode>::vsnray_type;

public:

    VSNRAY_FUNC cuda_texture_ref() = default;

    VSNRAY_CPU_FUNC cuda_texture_ref(cuda_texture<T, ReadMode, 3> const& ref)
        : texture_obj_(ref.texture_object())
        , width_(ref.width())
    {
    }

    VSNRAY_FUNC ~cuda_texture_ref() = default;

    VSNRAY_FUNC cuda_texture_ref& operator=(cuda_texture<T, ReadMode, 3> const& rhs)
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

private:

    cudaTextureObject_t texture_obj_;

    size_t width_;

};

} // visionaray
