// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_CUDA_TEXTURE_H
#define VSNRAY_TEXTURE_CUDA_TEXTURE_H 1

//-----------------------------------------------------------------------------
// this header is included by <texture/texture.h> _if_ compiling with nvcc
// (ifdef __CUDACC__), but can be safely included by the user, e.g., if
// not compiling with nvcc, but with the runtime API headers included
//-----------------------------------------------------------------------------

#include "detail/cuda_texture.h"

namespace visionaray
{

//-----------------------------------------------------------------------------
// the following function are host/device to allow for mixing with their host
// counterparts (for non-cuda types) in cross-platform code; they don't have a
// useful host implementation though!
//-----------------------------------------------------------------------------

template <
    typename T,
    cudaTextureReadMode ReadMode = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, 1>::cuda_type, ReadMode>::vsnray_return_type
tex1D(cuda_texture_ref<T, 1> const& tex, float coord)
{
#ifdef __CUDA_ARCH__
    using tex_type          = cuda_texture_ref<T, 1>;
    using cuda_type         = typename tex_type::cuda_type;
    using return_type       = typename cuda::map_texel_type<cuda_type, ReadMode>::vsnray_return_type;
    using cuda_return_type  = typename cuda::map_texel_type<cuda_type, ReadMode>::cuda_return_type;

    cuda_return_type retval;

    ::tex1D(
            &retval,
            tex.texture_object(),
            coord
            );

    return cuda::cast<return_type>(retval);
#else
    // ERROR
#endif
}

template <
    typename T,
    cudaTextureReadMode ReadMode = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, 2>::cuda_type, ReadMode>::vsnray_return_type
tex2D(cuda_texture_ref<T, 2> const& tex, vector<2, float> coord)
{
#ifdef __CUDA_ARCH__
    using tex_type          = cuda_texture_ref<T, 2>;
    using cuda_type         = typename tex_type::cuda_type;
    using return_type       = typename cuda::map_texel_type<cuda_type, ReadMode>::vsnray_return_type;
    using cuda_return_type  = typename cuda::map_texel_type<cuda_type, ReadMode>::cuda_return_type;

    cuda_return_type retval;

    ::tex2D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y
            );

    return cuda::cast<return_type>(retval);
#else
    // ERROR
#endif
}

template <
    typename T,
    cudaTextureReadMode ReadMode = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, 3>::cuda_type, ReadMode>::vsnray_return_type
tex3D(cuda_texture_ref<T, 3> const& tex, vector<3, float> coord)
{
#ifdef __CUDA_ARCH__
    using tex_type          = cuda_texture_ref<T, 3>;
    using cuda_type         = typename tex_type::cuda_type;
    using return_type       = typename cuda::map_texel_type<cuda_type, ReadMode>::vsnray_return_type;
    using cuda_return_type  = typename cuda::map_texel_type<cuda_type, ReadMode>::cuda_return_type;

    cuda_return_type retval;

    ::tex3D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y,
            coord.z
            );

    return cuda::cast<return_type>(retval);
#else
    // ERROR
#endif
}

} // visionaray

#endif // VSNRAY_TEXTURE_CUDA_TEXTURE_H
