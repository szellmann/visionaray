// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_HIP_TEXTURE_H
#define VSNRAY_TEXTURE_HIP_TEXTURE_H 1

//-----------------------------------------------------------------------------
// this header is included by <texture/texture.h> _if_ compiling with nvcc
// (ifdef __HIPCC__), but can be safely included by the user, e.g., if
// not compiling with nvcc, but with the runtime API headers included
//-----------------------------------------------------------------------------

#include "detail/hip_texture.h"

namespace visionaray
{

//-----------------------------------------------------------------------------
// the following function are host/device to allow for mixing with their host
// counterparts (for non-hip types) in cross-platform code; they don't have a
// useful host implementation though!
//-----------------------------------------------------------------------------

template <
    typename T,
    hipTextureReadMode ReadMode = hipTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename hip::map_texel_type<typename hip_texture_ref<T, 1>::hip_type, ReadMode>::vsnray_return_type
tex1D(hip_texture_ref<T, 1> const& tex, float coord)
{
#if __HIP_DEVICE_COMPILE__
    using tex_type          = hip_texture_ref<T, 1>;
    using hip_type         = typename tex_type::hip_type;
    using return_type       = typename hip::map_texel_type<hip_type, ReadMode>::vsnray_return_type;
    using hip_return_type  = typename hip::map_texel_type<hip_type, ReadMode>::hip_return_type;

    hip_return_type retval;

    ::tex1D(
            &retval,
            tex.texture_object(),
            coord
            );

    return hip::cast<return_type>(retval);
#else
    // ERROR
#endif
}

template <
    typename T,
    hipTextureReadMode ReadMode = hipTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename hip::map_texel_type<typename hip_texture_ref<T, 2>::hip_type, ReadMode>::vsnray_return_type
tex2D(hip_texture_ref<T, 2> const& tex, vector<2, float> coord)
{
#if __HIP_DEVICE_COMPILE__
    using tex_type          = hip_texture_ref<T, 2>;
    using hip_type         = typename tex_type::hip_type;
    using return_type       = typename hip::map_texel_type<hip_type, ReadMode>::vsnray_return_type;
    using hip_return_type  = typename hip::map_texel_type<hip_type, ReadMode>::hip_return_type;

    hip_return_type retval;

    ::tex2D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y
            );

    return hip::cast<return_type>(retval);
#else
    // ERROR
#endif
}

template <
    typename T,
    hipTextureReadMode ReadMode = hipTextureReadMode(detail::tex_read_mode_from_type<T>::value)
    >
VSNRAY_FUNC
inline typename hip::map_texel_type<typename hip_texture_ref<T, 3>::hip_type, ReadMode>::vsnray_return_type
tex3D(hip_texture_ref<T, 3> const& tex, vector<3, float> coord)
{
#if __HIP_DEVICE_COMPILE__
    using tex_type          = hip_texture_ref<T, 3>;
    using hip_type         = typename tex_type::hip_type;
    using return_type       = typename hip::map_texel_type<hip_type, ReadMode>::vsnray_return_type;
    using hip_return_type  = typename hip::map_texel_type<hip_type, ReadMode>::hip_return_type;

    hip_return_type retval;

    ::tex3D(
            &retval,
            tex.texture_object(),
            coord.x,
            coord.y,
            coord.z
            );

    return hip::cast<return_type>(retval);
#else
    // ERROR
#endif
}

} // visionaray

#endif // VSNRAY_TEXTURE_CUDA_TEXTURE_H
