// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_UTIL_H
#define VSNRAY_CUDA_UTIL_H 1

#include <cuda_runtime_api.h>

#include <visionaray/detail/macros.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>

#include "cast.h"

namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
// cuda texel type <-> visionaray texel type
//

template <typename Type, cudaTextureReadMode ReadMode>
struct map_texel_type
{
    using cuda_type             = Type;
    using vsnray_type           = Type;

    using cuda_return_type      = Type;
    using vsnray_return_type    = Type;
};


//-------------------------------------------------------------------------------------------------
// cuda -> visionaray
//

template <>
struct map_texel_type<uchar4, cudaReadModeElementType>
{
    using cuda_type             = uchar4;
    using vsnray_type           = vector<4, unsigned char>;

    using cuda_return_type      = uchar4;
    using vsnray_return_type    = vector<4, unsigned char>;
};

template <>
struct map_texel_type<unsigned char, cudaReadModeNormalizedFloat>
{
    using cuda_type             = unsigned char;
    using vsnray_type           = unorm<8>;

    using cuda_return_type      = float;
    using vsnray_return_type    = float;
};

template <>
struct map_texel_type<uchar2, cudaReadModeNormalizedFloat>
{
    using cuda_type             = uchar2;
    using vsnray_type           = vector<2, unorm<8>>;

    using cuda_return_type      = float2;
    using vsnray_return_type    = vector<2, float>;
};

template <>
struct map_texel_type<uchar3, cudaReadModeNormalizedFloat>
{
    using cuda_type             = uchar3;
    using vsnray_type           = vector<3, unorm<8>>;

    using cuda_return_type      = float3;
    using vsnray_return_type    = vector<3, float>;
};

template <>
struct map_texel_type<uchar4, cudaReadModeNormalizedFloat>
{
    using cuda_type             = uchar4;
    using vsnray_type           = vector<4, unorm<8>>;

    using cuda_return_type      = float4;
    using vsnray_return_type    = vector<4, float>;
};

template <>
struct map_texel_type<unsigned short, cudaReadModeNormalizedFloat>
{
    using cuda_type             = unsigned short;
    using vsnray_type           = unorm<16>;

    using cuda_return_type      = float;
    using vsnray_return_type    = float;
};

template <>
struct map_texel_type<ushort2, cudaReadModeNormalizedFloat>
{
    using cuda_type             = ushort2;
    using vsnray_type           = vector<2, unorm<16>>;

    using cuda_return_type      = float2;
    using vsnray_return_type    = vector<2, float>;
};

template <>
struct map_texel_type<ushort3, cudaReadModeNormalizedFloat>
{
    using cuda_type             = ushort3;
    using vsnray_type           = vector<3, unorm<16>>;

    using cuda_return_type      = float3;
    using vsnray_return_type    = vector<3, float>;
};

template <>
struct map_texel_type<ushort4, cudaReadModeNormalizedFloat>
{
    using cuda_type             = ushort4;
    using vsnray_type           = vector<4, unorm<16>>;

    using cuda_return_type      = float4;
    using vsnray_return_type    = vector<4, float>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<float2, ReadMode>
{
    using cuda_type             = float2;
    using vsnray_type           = vector<2, float>;

    using cuda_return_type      = float2;
    using vsnray_return_type    = vector<2, float>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<float4, ReadMode>
{
    using cuda_type             = float4;
    using vsnray_type           = vector<4, float>;

    using cuda_return_type      = float4;
    using vsnray_return_type    = vector<4, float>;
};


//-------------------------------------------------------------------------------------------------
// cuda <- visionaray
//

template <cudaTextureReadMode ReadMode>
struct map_texel_type<unorm<8>, ReadMode>
{
    using cuda_type   = unsigned char;
    using vsnray_type = unorm<8>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<2, unorm<8>>, ReadMode>
{
    using cuda_type   = uchar2;
    using vsnray_type = vector<2, unorm<8>>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<3, unorm<8>>, ReadMode>
{
    using cuda_type   = uchar3;
    using vsnray_type = vector<3, unorm<8>>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<4, unorm<8>>, ReadMode>
{
    using cuda_type   = uchar4;
    using vsnray_type = vector<4, unorm<8>>;

    using cuda_return_type   = float4;
    using vsnray_return_type = vector<4, unorm<8>>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<unorm<16>, ReadMode>
{
    using cuda_type   = unsigned short;
    using vsnray_type = unorm<16>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<2, unorm<16>>, ReadMode>
{
    using cuda_type   = ushort2;
    using vsnray_type = vector<2, unorm<16>>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<3, unorm<16>>, ReadMode>
{
    using cuda_type   = ushort3;
    using vsnray_type = vector<3, unorm<16>>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<4, unorm<16>>, ReadMode>
{
    using cuda_type   = ushort4;
    using vsnray_type = vector<4, unorm<16>>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<2, float>, ReadMode>
{
    using cuda_type   = float2;
    using vsnray_type = vector<2, float>;
};

template <cudaTextureReadMode ReadMode>
struct map_texel_type<vector<4, float>, ReadMode>
{
    using cuda_type   = float4;
    using vsnray_type = vector<4, float>;

    using cuda_return_type   = float4;
    using vsnray_return_type = vector<4, float>;
};

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_UTIL_H
