// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/cuda/array.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// CUDA texture1d
//

template <typename T, tex_read_mode ReadMode>
class cuda_texture<T, ReadMode, 1>
{
public:

    using cuda_type   = typename cuda::map_texel_type<T, ReadMode>::cuda_type;
    using vsnray_type = typename cuda::map_texel_type<T, ReadMode>::vsnray_type;

public:

    cuda_texture() = default;

    // Construct from host texture
    template <typename U>
    explicit cuda_texture(texture<U, ReadMode, 1> const& host_tex)
        : width_(host_tex.width())
    {
        if (width_ == 0)
        {
            return;
        }


        cudaChannelFormatDesc desc = cudaCreateChannelDesc<cuda_type>();

        if ( array_.allocate(desc, width_) != cudaSuccess )
        {
            return;
        }

        if ( upload_data(host_tex.data()) != cudaSuccess )
        {
            return;
        }

        cudaResourceDesc resource_desc;
        memset(&resource_desc, 0, sizeof(resource_desc));
        resource_desc.resType                   = cudaResourceTypeArray;
        resource_desc.res.array.array           = array_.get();

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0]             = detail::map_address_mode( host_tex.get_address_mode(0) );
        texture_desc.filterMode                 = detail::map_filter_mode( host_tex.get_filter_mode() );
        texture_desc.readMode                   = detail::map_read_mode( ReadMode );
        texture_desc.normalizedCoords           = true;

        cudaTextureObject_t obj = 0;
        cudaCreateTextureObject( &obj, &resource_desc, &texture_desc, 0 );
        texture_obj_.reset(obj);
    }

#if !VSNRAY_CXX_MSVC
    cuda_texture(cuda_texture&&) = default;
    cuda_texture& operator=(cuda_texture&&) = default;
#else
    cuda_texture(cuda_texture&& rhs)
        : buffer_(std::move(rhs.buffer_))
        , texture_obj_(std::move(rhs.texture_obj_))
        , width_(rhs.width_)
    {
    }

    cuda_texture& operator=(cuda_texture&& rhs)
    {
        buffer_ = std::move(rhs.buffer_);
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

private:

    cuda::array             array_;

    cuda::texture_object    texture_obj_;

    size_t width_;

    cudaError_t upload_data(vsnray_type const* data)
    {
        // Cast from host type to device type
        return array_.upload( reinterpret_cast<cuda_type const*>(data), width_ * sizeof(cuda_type) );
    }

    template <typename U>
    cudaError_t upload_data(U const* data)
    {
        // First promote to host type
        aligned_vector<vsnray_type> dst( width_ );

        for (size_t i = 0; i < width_; ++i)
        {
            dst[i] = vsnray_type( data[i] );
        }

        return upload_data( dst.data() );
    }

};


//-------------------------------------------------------------------------------------------------
// CUDA texture1d reference
//

template <typename T, tex_read_mode ReadMode>
class cuda_texture_ref<T, ReadMode, 1>
{
public:

    using cuda_type   = typename cuda::map_texel_type<T, ReadMode>::cuda_type;
    using vsnray_type = typename cuda::map_texel_type<T, ReadMode>::vsnray_type;

public:

    VSNRAY_FUNC cuda_texture_ref() = default;

    VSNRAY_CPU_FUNC cuda_texture_ref(cuda_texture<T, ReadMode, 1> const& ref)
        : texture_obj_(ref.texture_object())
        , width_(ref.width())
    {
    }

    VSNRAY_FUNC ~cuda_texture_ref() = default;

    VSNRAY_FUNC cuda_texture_ref& operator=(cuda_texture<T, ReadMode, 1> const& rhs)
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
