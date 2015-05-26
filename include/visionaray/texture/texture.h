// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_H
#define VSNRAY_TEXTURE_H

#include <visionaray/detail/macros.h>

#ifdef __CUDACC__
#include "detail/device_texture.h"
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

//-------------------------------------------------------------------------------------------------
// tex1D - general case and specializations
//

template <typename Tex, typename FloatT>
inline typename Tex::value_type tex1D(Tex const& tex, FloatT coord)
{

    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    // general case: return type equals texel type
    typedef typename Tex::value_type return_type;

    return detail::tex1D<return_type>( tex, coord );

}


template <typename Tex>
inline vector<4, simd::float4> tex1D(Tex const& tex, simd::float4 coord)
{

    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    // special case for AoS rgba colors
    typedef vector<4, simd::float4> return_type;

    return detail::tex1D<return_type>( tex, coord );

}


//-------------------------------------------------------------------------------------------------
// tex2D - general case and specializations
//

template <typename Tex, typename FloatT>
inline typename Tex::value_type tex2D(Tex const& tex, vector<2, FloatT> coord)
{
    static_assert(Tex::dimensions == 2, "Incompatible texture type");

    using return_type = typename Tex::value_type;

    return detail::tex2D<return_type>( tex, coord );
}


#ifdef __CUDACC__

template <typename T, tex_read_mode ReadMode>
VSNRAY_GPU_FUNC
inline T tex2D(device_texture_ref<T, ReadMode, 2> const& tex, vector<2, float> coord)
{
    using tex_type    = device_texture_ref<T, ReadMode, 2>;
    using device_type = typename tex_type::device_type;

    return cuda::map_texel_type<device_type>::cast( ::tex2D<device_type>(
                tex.texture_object(),
                coord.x,
                coord.y
                ) );
}

#endif // __CUDACC__


//-------------------------------------------------------------------------------------------------
// tex3D - general case and specializations
//

template <typename Tex, typename FloatT>
inline typename Tex::value_type tex3D(Tex const& tex, vector<3, FloatT> coord)
{

    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    // general case: return type equals texel type
    typedef typename Tex::value_type return_type;

    return detail::tex3D<return_type>( tex, coord );

}


template <typename Tex>
inline simd::float4 tex3D(Tex const& tex, vector<3, simd::float4> coord)
{

    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    // special case: lookup four texels at once and return as 32-bit float vector
    typedef simd::float4 return_type;

    return detail::tex3D<return_type>( tex, coord );

}


} // visionaray


#endif // VSNRAY_TEXTURE_H
