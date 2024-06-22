// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_HIP_TEXTURE_H
#define VSNRAY_TEXTURE_DETAIL_HIP_TEXTURE_H 1

#include <cstddef>

#include "../../hip/util.h"
#include "../../detail/macros.h"
#include "../../math/unorm.h"
#include "../../math/vector.h"

#include "texture_common.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Map visionaray texture address mode to hip texture address mode
//

inline hipTextureAddressMode map_address_mode(tex_address_mode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case Wrap:
        return hipAddressModeWrap;

    case Mirror:
        return hipAddressModeMirror;

    case Clamp:
        return hipAddressModeClamp;

    case Border:
        return hipAddressModeBorder;

    }
}


//-------------------------------------------------------------------------------------------------
// Map visionaray texture filter mode to hip texture filter mode
//

inline hipTextureFilterMode map_filter_mode(tex_filter_mode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case Nearest:
        return hipFilterModePoint;

    case Linear:
        return hipFilterModeLinear;

    }
}


//-------------------------------------------------------------------------------------------------
// Deduce texture read mode from type
//

template <typename T>
struct tex_read_mode_from_type
{
    enum { value = hipReadModeElementType };
};

template <unsigned Bits>
struct tex_read_mode_from_type<unorm<Bits>>
{
    enum { value = hipReadModeNormalizedFloat };
};

template <size_t Dim, unsigned Bits>
struct tex_read_mode_from_type<vector<Dim, unorm<Bits>>>
{
    enum { value = hipReadModeNormalizedFloat };
};

} // detail


//-------------------------------------------------------------------------------------------------
// Forward-declare hip texture template
//

template <typename T, unsigned Dim>
class hip_texture;

template <typename T, unsigned Dim>
class hip_texture_ref;

} // visionaray

#include "hip_texture1d.inl"
#include "hip_texture2d.inl"
#include "hip_texture3d.inl"

#endif // VSNRAY_TEXTURE_DETAIL_HIP_TEXTURE_H
