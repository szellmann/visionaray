// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_H
#define VSNRAY_TEXTURE_H 1

#include <cassert>

#include "../detail/macros.h"

#ifdef __CUDACC__
#include "detail/cuda_texture.h"
#endif

#include "detail/tex_fetch.h"
#include "detail/texture_common.h"

namespace visionaray
{

template <typename Tex, typename FloatT>
VSNRAY_FUNC
inline auto tex1D(Tex const& tex, FloatT const& coord)
    -> decltype( detail::tex_fetch_impl(tex, vector<1, FloatT>(coord)) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    assert(tex.get_normalized_coords() && "Unnormalized coordinates on CPU not implemented yet");

    return detail::tex_fetch_impl( tex, vector<1, FloatT>(coord) );
}


template <typename Tex, typename FloatT>
VSNRAY_FUNC
inline auto tex2D(Tex const& tex, vector<2, FloatT> const& coord)
    -> decltype( detail::tex_fetch_impl(tex, coord) )
{
    static_assert(Tex::dimensions == 2, "Incompatible texture type");

    assert(tex.get_normalized_coords() && "Unnormalized coordinates on CPU not implemented yet");

    return detail::tex_fetch_impl( tex, coord );
}


template <typename Tex, typename FloatT>
VSNRAY_FUNC
inline auto tex3D(Tex const& tex, vector<3, FloatT> const& coord)
    -> decltype( detail::tex_fetch_impl(tex, coord) )
{
    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    assert(tex.get_normalized_coords() && "Unnormalized coordinates on CPU not implemented yet");

    return detail::tex_fetch_impl( tex, coord );
}


} // visionaray

#ifdef __CUDACC__
#include "cuda_texture.h"
#endif // __CUDACC__

#endif // VSNRAY_TEXTURE_H
