// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_TEXTURE_H
#define VSNRAY_CUDA_TEXTURE_H 1

#include <cstddef>
#include <cstring> // memset

#include <array>

#include <visionaray/cuda/pitch2d.h>
#include <visionaray/cuda/texture_object.h>
#include <visionaray/cuda/util.h>
#include <visionaray/detail/macros.h>

#include "texture_common.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Map visionaray texture address mode to cuda texture address mode
//

cudaTextureAddressMode map_address_mode(tex_address_mode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case Wrap:
        return cudaAddressModeWrap;

    case Mirror:
        return cudaAddressModeMirror;

    case Clamp:
        return cudaAddressModeClamp;

    case Border:
        return cudaAddressModeBorder;

    }
}


//-------------------------------------------------------------------------------------------------
// Map visionaray texture filter mode to cuda texture filter mode
//

cudaTextureFilterMode map_filter_mode(tex_filter_mode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case Nearest:
        return cudaFilterModePoint;

    case Linear:
        return cudaFilterModeLinear;

    }
}


//-------------------------------------------------------------------------------------------------
// Map visionaray texture read mode to cuda texture read mode
//

cudaTextureReadMode map_read_mode(tex_read_mode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case ElementType:
        return cudaReadModeElementType;

    case NormalizedFloat:
        return cudaReadModeNormalizedFloat;

    }
}

} // detail

//-------------------------------------------------------------------------------------------------
// CUDA texture
//

template <typename T, tex_read_mode ReadMode, size_t Dim>
class cuda_texture;

template <typename T, tex_read_mode ReadMode>
class cuda_texture<T, ReadMode, 2>
{
public:

    using device_type = typename cuda::map_texel_type<T, ReadMode>::device_type;
    using host_type   = typename cuda::map_texel_type<T, ReadMode>::host_type;

public:

    cuda_texture() = default;

    // Construct from host texture
    template <typename U>
    explicit cuda_texture(texture<U, ReadMode, 2> const& host_tex)
        : width_(host_tex.width())
        , height_(host_tex.height())
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

        auto desc = cudaCreateChannelDesc<device_type>();

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
        texture_desc.addressMode[0]             = detail::map_address_mode( host_tex.get_address_mode(0) );
        texture_desc.addressMode[1]             = detail::map_address_mode( host_tex.get_address_mode(1) );
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

private:

    cuda::pitch2d<device_type>  pitch_;

    cuda::texture_object        texture_obj_;

    size_t width_;
    size_t height_;

    cudaError_t upload_data(host_type const* data)
    {
        // Cast from host type to device type
        return pitch_.upload( reinterpret_cast<device_type const*>(data), width_, height_ );
    }

    template <typename U>
    cudaError_t upload_data(U const* data)
    {
        // First promote to host type
        aligned_vector<host_type> dst( width_ * height_ );

        for (size_t i = 0; i < width_ * height_; ++i)
        {
            dst[i] = host_type( data[i] );
        }

        return upload_data( dst.data() );
    }

};


//-------------------------------------------------------------------------------------------------
// CUDA texture reference
//

template <typename T, tex_read_mode ReadMode, size_t Dim>
class cuda_texture_ref
{
public:

    using device_type = typename cuda::map_texel_type<T, ReadMode>::device_type;
    using host_type   = typename cuda::map_texel_type<T, ReadMode>::host_type;

public:

    VSNRAY_FUNC cuda_texture_ref() = default;

    VSNRAY_CPU_FUNC cuda_texture_ref(cuda_texture<T, ReadMode, Dim> const& ref)
        : texture_obj_(ref.texture_object())
        , width_(ref.width())
        , height_(ref.height())
    {
    }

    VSNRAY_FUNC ~cuda_texture_ref() = default;

    VSNRAY_FUNC cuda_texture_ref& operator=(cuda_texture<T, ReadMode, Dim> const& rhs)
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

private:

    cudaTextureObject_t texture_obj_;

    size_t width_;
    size_t height_;

};

} // visionaray

#endif // VSNRAY_CUDA_TEXTURE_H
