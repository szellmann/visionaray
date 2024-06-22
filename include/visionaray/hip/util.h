// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_UTIL_H
#define VSNRAY_HIP_UTIL_H 1

#include <hip/hip_runtime_api.h>

#include <visionaray/detail/macros.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>

#include "cast.h"

namespace visionaray
{
namespace hip
{

//-------------------------------------------------------------------------------------------------
// hip texel type <-> visionaray texel type
//

template <typename Type, hipTextureReadMode ReadMode>
struct map_texel_type
{
    using hip_type              = Type;
    using vsnray_type           = Type;

    using hip_return_type       = Type;
    using vsnray_return_type    = Type;
};


//-------------------------------------------------------------------------------------------------
// hip -> visionaray
//

template <>
struct map_texel_type<uchar4, hipReadModeElementType>
{
    using hip_type             = uchar4;
    using vsnray_type           = vector<4, unsigned char>;

    using hip_return_type      = uchar4;
    using vsnray_return_type    = vector<4, unsigned char>;
};

template <>
struct map_texel_type<unsigned char, hipReadModeNormalizedFloat>
{
    using hip_type             = unsigned char;
    using vsnray_type           = unorm<8>;

    using hip_return_type      = float;
    using vsnray_return_type    = float;
};

template <>
struct map_texel_type<uchar2, hipReadModeNormalizedFloat>
{
    using hip_type             = uchar2;
    using vsnray_type           = vector<2, unorm<8>>;

    using hip_return_type      = float2;
    using vsnray_return_type    = vector<2, float>;
};

template <>
struct map_texel_type<uchar3, hipReadModeNormalizedFloat>
{
    using hip_type             = uchar3;
    using vsnray_type           = vector<3, unorm<8>>;

    using hip_return_type      = float3;
    using vsnray_return_type    = vector<3, float>;
};

template <>
struct map_texel_type<uchar4, hipReadModeNormalizedFloat>
{
    using hip_type             = uchar4;
    using vsnray_type           = vector<4, unorm<8>>;

    using hip_return_type      = float4;
    using vsnray_return_type    = vector<4, float>;
};

template <>
struct map_texel_type<unsigned short, hipReadModeNormalizedFloat>
{
    using hip_type             = unsigned short;
    using vsnray_type           = unorm<16>;

    using hip_return_type      = float;
    using vsnray_return_type    = float;
};

template <>
struct map_texel_type<ushort2, hipReadModeNormalizedFloat>
{
    using hip_type             = ushort2;
    using vsnray_type           = vector<2, unorm<16>>;

    using hip_return_type      = float2;
    using vsnray_return_type    = vector<2, float>;
};

template <>
struct map_texel_type<ushort3, hipReadModeNormalizedFloat>
{
    using hip_type             = ushort3;
    using vsnray_type           = vector<3, unorm<16>>;

    using hip_return_type      = float3;
    using vsnray_return_type    = vector<3, float>;
};

template <>
struct map_texel_type<ushort4, hipReadModeNormalizedFloat>
{
    using hip_type             = ushort4;
    using vsnray_type           = vector<4, unorm<16>>;

    using hip_return_type      = float4;
    using vsnray_return_type    = vector<4, float>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<float2, ReadMode>
{
    using hip_type             = float2;
    using vsnray_type           = vector<2, float>;

    using hip_return_type      = float2;
    using vsnray_return_type    = vector<2, float>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<float4, ReadMode>
{
    using hip_type             = float4;
    using vsnray_type           = vector<4, float>;

    using hip_return_type      = float4;
    using vsnray_return_type    = vector<4, float>;
};


//-------------------------------------------------------------------------------------------------
// hip <- visionaray
//

template <hipTextureReadMode ReadMode>
struct map_texel_type<unorm<8>, ReadMode>
{
    using hip_type   = unsigned char;
    using vsnray_type = unorm<8>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<2, unorm<8>>, ReadMode>
{
    using hip_type   = uchar2;
    using vsnray_type = vector<2, unorm<8>>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<3, unorm<8>>, ReadMode>
{
    using hip_type   = uchar3;
    using vsnray_type = vector<3, unorm<8>>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<4, unorm<8>>, ReadMode>
{
    using hip_type   = uchar4;
    using vsnray_type = vector<4, unorm<8>>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<unorm<16>, ReadMode>
{
    using hip_type   = unsigned short;
    using vsnray_type = unorm<16>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<2, unorm<16>>, ReadMode>
{
    using hip_type   = ushort2;
    using vsnray_type = vector<2, unorm<16>>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<3, unorm<16>>, ReadMode>
{
    using hip_type   = ushort3;
    using vsnray_type = vector<3, unorm<16>>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<4, unorm<16>>, ReadMode>
{
    using hip_type   = ushort4;
    using vsnray_type = vector<4, unorm<16>>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<2, float>, ReadMode>
{
    using hip_type   = float2;
    using vsnray_type = vector<2, float>;
};

template <hipTextureReadMode ReadMode>
struct map_texel_type<vector<4, float>, ReadMode>
{
    using hip_type   = float4;
    using vsnray_type = vector<4, float>;
};

} // hip
} // visionaray

#endif // VSNRAY_HIP_UTIL_H
