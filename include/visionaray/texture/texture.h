// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_H
#define VSNRAY_TEXTURE_H 1

#include <visionaray/detail/macros.h>

#if defined(__CUDACC__) || defined(CUDARTAPI)
#include "detail/cuda_texture.h"
#endif

#include "detail/prefilter.h"
#include "detail/sampler1d.h"
#include "detail/sampler2d.h"
#include "detail/sampler3d.h"
#include "detail/texture1d.h"
#include "detail/texture2d.h"
#include "detail/texture3d.h"


namespace visionaray
{

template <typename Tex, typename FloatT>
inline auto tex1D(Tex const& tex, FloatT const& coord)
    -> decltype( detail::tex1D_impl(tex, coord) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    assert(tex.get_normalized_coords() && "Unnormalized coordinates on CPU not implemented yet");

    return detail::tex1D_impl( tex, coord );
}


template <typename Tex, typename FloatT>
inline auto tex2D(Tex const& tex, vector<2, FloatT> const& coord)
    -> decltype( detail::tex2D_impl(tex, coord) )
{
    static_assert(Tex::dimensions == 2, "Incompatible texture type");

    assert(tex.get_normalized_coords() && "Unnormalized coordinates on CPU not implemented yet");

    return detail::tex2D_impl( tex, coord );
}


template <typename Tex, typename FloatT>
inline auto tex3D(Tex const& tex, vector<3, FloatT> const& coord)
    -> decltype( detail::tex3D_impl(tex, coord) )
{
    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    assert(tex.get_normalized_coords() && "Unnormalized coordinates on CPU not implemented yet");

    return detail::tex3D_impl( tex, coord );
}


#if defined(__CUDACC__) || defined(CUDARTAPI)

template <
    typename T,
    cudaTextureReadMode ReadMode = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, 1>::cuda_type, ReadMode>::vsnray_return_type
tex1D(cuda_texture_ref<T, 1> const& tex, float coord)
{
    using tex_type          = cuda_texture_ref<T, 1>;
    using cuda_type         = typename tex_type::cuda_type;
    using return_type       = typename cuda::map_texel_type<cuda_type, ReadMode>::vsnray_return_type;
    using cuda_return_type  = typename cuda::map_texel_type<cuda_type, ReadMode>::cuda_return_type;

    cuda_return_type retval;

#if VSNRAY_GPU_MODE
    ::tex1D(
            &retval,
            tex.texture_object(),
            coord
            );
#endif

    return cuda::cast<return_type>(retval);
}

template <
    typename T,
    cudaTextureReadMode ReadMode = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, 2>::cuda_type, ReadMode>::vsnray_return_type
tex2D(cuda_texture_ref<T, 2> const& tex, vector<2, float> coord)
{
    using tex_type          = cuda_texture_ref<T, 2>;
    using cuda_type         = typename tex_type::cuda_type;
    using return_type       = typename cuda::map_texel_type<cuda_type, ReadMode>::vsnray_return_type;
    using cuda_return_type  = typename cuda::map_texel_type<cuda_type, ReadMode>::cuda_return_type;

    cuda_return_type retval;

#if VSNRAY_GPU_MODE
    ::tex2D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y
            );
#endif

    return cuda::cast<return_type>(retval);
}

template <
    typename T,
    cudaTextureReadMode ReadMode = cudaTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, 3>::cuda_type, ReadMode>::vsnray_return_type
tex3D(cuda_texture_ref<T, 3> const& tex, vector<3, float> coord)
{
    using tex_type          = cuda_texture_ref<T, 3>;
    using cuda_type         = typename tex_type::cuda_type;
    using return_type       = typename cuda::map_texel_type<cuda_type, ReadMode>::vsnray_return_type;
    using cuda_return_type  = typename cuda::map_texel_type<cuda_type, ReadMode>::cuda_return_type;

    cuda_return_type retval;

#if VSNRAY_GPU_MODE
    ::tex3D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y,
            coord.z
            );
#endif

    return cuda::cast<return_type>(retval);
}

#endif // defined(__CUDACC__) || defined(CUDARTAPI)

} // visionaray

#endif // VSNRAY_TEXTURE_H
