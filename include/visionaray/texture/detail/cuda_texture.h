// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_CUDA_TEXTURE_H
#define VSNRAY_TEXTURE_DETAIL_CUDA_TEXTURE_H 1

#include <cstddef>
#include <cstring> // memset

#include <array>

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
// Forward declarations
//

template <typename T, tex_read_mode ReadMode, size_t Dim>
class cuda_texture;

template <typename T, tex_read_mode ReadMode, size_t Dim>
class cuda_texture_ref;

} // visionaray

#include "cuda_texture1d.inl"
#include "cuda_texture2d.inl"
#include "cuda_texture3d.inl"

#endif // VSNRAY_TEXTURE_DETAIL_CUDA_TEXTURE_H
