// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_CUDA_TEXTURE_H
#define VSNRAY_TEXTURE_DETAIL_CUDA_TEXTURE_H 1

#include <cstddef>

#include "../../cuda/util.h"
#include "../../detail/macros.h"
#include "../../math/unorm.h"
#include "../../math/vector.h"

#include "texture_common.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Map visionaray texture address mode to cuda texture address mode
//

inline cudaTextureAddressMode map_address_mode(tex_address_mode mode)
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

inline cudaTextureFilterMode map_filter_mode(tex_filter_mode mode)
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
// Deduce texture read mode from type
//

template <typename T>
struct tex_read_mode_from_type
{
    enum { value = cudaReadModeElementType };
};

template <unsigned Bits>
struct tex_read_mode_from_type<unorm<Bits>>
{
    enum { value = cudaReadModeNormalizedFloat };
};

template <size_t Dim, unsigned Bits>
struct tex_read_mode_from_type<vector<Dim, unorm<Bits>>>
{
    enum { value = cudaReadModeNormalizedFloat };
};

} // detail


//-------------------------------------------------------------------------------------------------
// Forward-declare cuda texture template
//

template <typename T, unsigned Dim>
class cuda_texture;

template <typename T, unsigned Dim>
class cuda_texture_ref;

} // visionaray

#include "cuda_texture1d.inl"
#include "cuda_texture2d.inl"
#include "cuda_texture3d.inl"

#endif // VSNRAY_TEXTURE_DETAIL_CUDA_TEXTURE_H
