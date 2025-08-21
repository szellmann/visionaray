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

VSNRAY_FUNC
inline float tex1D(cuda_texture_ref<float, 1> const& tex, float coord)
{
#ifdef __CUDA_ARCH__
    float retval;

    ::tex1D(
            &retval,
            tex.texture_object(),
            coord
            );

    return retval;
#else
    // ERROR
    return {};
#endif
}

template <typename T>
VSNRAY_FUNC
vector<4, float> tex1D(cuda_texture_ref<vector<4, T>, 1> const& tex, float coord)
{
#ifdef __CUDA_ARCH__
    ::float4 retval;

    ::tex1D(
            &retval,
            tex.texture_object(),
            coord
            );

    return cuda::cast<vector<4, float>>(retval);
#else
    // ERROR
    return {};
#endif
}

VSNRAY_FUNC
inline float tex2D(cuda_texture_ref<float, 2> const& tex, vector<2, float> coord)
{
#ifdef __CUDA_ARCH__
    float retval;

    ::tex2D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y
            );

    return retval;
#else
    // ERROR
    return {};
#endif
}

template <typename T>
VSNRAY_FUNC
vector<4, float> tex2D(cuda_texture_ref<vector<4, T>, 2> const& tex, vector<2, float> coord)
{
#ifdef __CUDA_ARCH__
    ::float4 retval;

    ::tex2D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y
            );

    return cuda::cast<vector<4, float>>(retval);
#else
    // ERROR
    return {};
#endif
}

VSNRAY_FUNC
inline float tex3D(cuda_texture_ref<float, 3> const& tex, vector<3, float> coord)
{
#ifdef __CUDA_ARCH__
    float retval;

    ::tex3D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y,
            coord.z
            );

    return retval;
#else
    // ERROR
    return {};
#endif
}

template <typename T>
VSNRAY_FUNC
vector<4, float> tex3D(cuda_texture_ref<vector<4, T>, 3> const& tex, vector<3, float> coord)
{
#ifdef __CUDA_ARCH__
    ::float4 retval;

    ::tex3D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y,
            coord.z
            );

    return cuda::cast<vector<4, float>>(retval);
#else
    // ERROR
    return {};
#endif
}

} // visionaray

#endif // VSNRAY_TEXTURE_CUDA_TEXTURE_H
