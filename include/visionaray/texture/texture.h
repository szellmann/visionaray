// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_H
#define VSNRAY_TEXTURE_H


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

template <typename TexelT, typename FloatT>
inline TexelT tex1D(texture_ref<TexelT, ElementType, 1> const& tex, FloatT coord)
{

    // general case: return type equals texel type
    typedef TexelT return_type;

    return detail::tex1D<return_type>( tex, coord );

}


template <typename TexelT>
inline vector<4, simd::float4> tex1D(texture_ref<TexelT, ElementType, 1> const& tex, simd::float4 coord)
{

    // special case for AoS rgba colors
    typedef vector<4, simd::float4> return_type;

    return detail::tex1D<return_type>( tex, coord );

}


//-------------------------------------------------------------------------------------------------
// tex2D - general case and specializations
//

template <typename TexelT, typename FloatT>
inline TexelT tex2D(texture_ref<TexelT, ElementType, 2> const& tex, vector<2, FloatT> coord)
{

    // general case: return type equals texel type
    typedef TexelT return_type;

    return detail::tex2D<return_type>( tex, coord );

}


template <typename TexelT>
inline vector<3, simd::float4> tex2D(texture_ref<TexelT, ElementType, 2> const& tex, vector<2, simd::float4> coord)
{

    // special case: lookup four texels at once and return as 32-bit float vector
    typedef vector<3, simd::float4> return_type;

    return detail::tex2D<return_type>( tex, coord );

}


//-------------------------------------------------------------------------------------------------
// tex3D - general case and specializations
//

template <typename TexelT, typename FloatT>
inline TexelT tex3D(texture_ref<TexelT, ElementType, 3> const& tex, vector<3, FloatT> coord)
{

    // general case: return type equals texel type
    typedef TexelT return_type;

    return detail::tex3D<return_type>( tex, coord );

}


template <typename TexelT>
inline simd::float4 tex3D(texture_ref<TexelT, ElementType, 3> const& tex, vector<3, simd::float4> coord)
{

    // special case: lookup four texels at once and return as 32-bit float vector
    typedef simd::float4 return_type;

    return detail::tex3D<return_type>( tex, coord );

}


} // visionaray


#endif // VSNRAY_TEXTURE_H
