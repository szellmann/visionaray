// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_H
#define VSNRAY_TEXTURE_H

#include <visionaray/detail/macros.h>

#ifdef __CUDACC__
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
inline auto tex1D(Tex const& tex, FloatT coord)
    -> decltype( detail::tex1D(tex, coord) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    return detail::tex1D( tex, coord );
}


template <typename Tex, typename FloatT>
inline auto tex2D(Tex const& tex, vector<2, FloatT> coord)
    -> decltype( detail::tex2D(tex, coord) )
{
    static_assert(Tex::dimensions == 2, "Incompatible texture type");

    return detail::tex2D( tex, coord );
}


template <typename Tex, typename FloatT>
inline auto tex3D(Tex const& tex, vector<3, FloatT> coord)
    -> decltype( detail::tex3D(tex, coord) )
{
    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    return detail::tex3D( tex, coord );
}


#ifdef __CUDACC__

template <typename T, tex_read_mode ReadMode>
VSNRAY_GPU_FUNC
inline typename cuda::map_texel_type<typename cuda_texture_ref<T, ReadMode, 2>::device_type, ReadMode>::host_return_type
tex2D(cuda_texture_ref<T, ReadMode, 2> const& tex, vector<2, float> coord)
{
    using tex_type              = cuda_texture_ref<T, ReadMode, 2>;
    using device_type           = typename tex_type::device_type;
    using device_return_type    = typename cuda::map_texel_type<device_type, ReadMode>::device_return_type;

    device_return_type retval;

    ::tex2D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y
            );

    return cuda::map_texel_type<device_type, ReadMode>::cast( retval );
}

#endif // __CUDACC__


} // visionaray


#endif // VSNRAY_TEXTURE_H
