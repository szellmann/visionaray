// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_UTIL_H
#define VSNRAY_CUDA_UTIL_H

#include <cuda_runtime_api.h>

#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/forward.h>

namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
// cuda texel type <-> visionaray texel type
//

template <typename Type, tex_read_mode ReadMode>
struct map_texel_type
{
    using device_type           = Type;
    using host_type             = Type;

    using device_return_type    = Type;
    using host_return_type      = Type;
};


//-------------------------------------------------------------------------------------------------
// cuda -> visionaray
//

template <>
struct map_texel_type<uchar4, ElementType>
{
    using device_type           = uchar4;
    using host_type             = vector<4, unorm<8>>;

    using device_return_type    = uchar4;
    using host_return_type      = vector<4, unorm<8>>;

    VSNRAY_FUNC static vector<4, unorm<8>> cast(uchar4 const& value)
    {
        return vector<4, unorm<8>>( value.x, value.y, value.z, value.w );
    }
};

template <>
struct map_texel_type<unsigned char, NormalizedFloat>
{
    using device_type           = unsigned char;
    using host_type             = unorm<8>;

    using device_return_type    = float;
    using host_return_type      = float;

    VSNRAY_FUNC static float cast(float value)
    {
        return value;
    }
};

template <>
struct map_texel_type<uchar2, NormalizedFloat>
{
    using device_type           = uchar2;
    using host_type             = vector<2, unorm<8>>;

    using device_return_type    = float2;
    using host_return_type      = vector<2, float>;

    VSNRAY_FUNC static vector<2, float> cast(float2 const& value)
    {
        return vector<2, float>( value.x, value.y );
    }
};

template <>
struct map_texel_type<uchar3, NormalizedFloat>
{
    using device_type           = uchar3;
    using host_type             = vector<3, unorm<8>>;

    using device_return_type    = float3;
    using host_return_type      = vector<3, float>;

    VSNRAY_FUNC static vector<3, float> cast(float3 const& value)
    {
        return vector<3, float>( value.x, value.y, value.z );
    }
};

template <>
struct map_texel_type<uchar4, NormalizedFloat>
{
    using device_type           = uchar4;
    using host_type             = vector<4, unorm<8>>;

    using device_return_type    = float4;
    using host_return_type      = vector<4, float>;

    VSNRAY_FUNC static vector<4, float> cast(float4 const& value)
    {
        return vector<4, float>( value.x, value.y, value.z, value.w );
    }
};

template <tex_read_mode ReadMode>
struct map_texel_type<float4, ReadMode>
{
    using device_type           = float4;
    using host_type             = vector<4, float>;

    using device_return_type    = float4;
    using host_return_type      = vector<4, float>;

    VSNRAY_FUNC static vector<4, float> cast(float4 const& value)
    {
        return vector<4, float>( value.x, value.y, value.z, value.w );
    }
};


//-------------------------------------------------------------------------------------------------
// cuda <- visionaray
//

template <tex_read_mode ReadMode>
struct map_texel_type<unorm<8>, ReadMode>
{
    using device_type = unsigned char;
    using host_type   = unorm<8>;

    VSNRAY_FUNC static unsigned char cast(unorm<8> value)
    {
        return static_cast<unsigned char>(value);
    }
};

template <tex_read_mode ReadMode>
struct map_texel_type<vector<2, unorm<8>>, ReadMode>
{
    using device_type = uchar2;
    using host_type   = vector<2, unorm<8>>;

    VSNRAY_FUNC static uchar2 cast(vector<2, unorm<8>> const& value)
    {
        return make_uchar2(
                static_cast<unsigned char>(value.x),
                static_cast<unsigned char>(value.y)
                );
    }
};

template <tex_read_mode ReadMode>
struct map_texel_type<vector<3, unorm<8>>, ReadMode>
{
    using device_type = uchar3;
    using host_type   = vector<3, unorm<8>>;

    VSNRAY_FUNC static uchar3 cast(vector<3, unorm<8>> const& value)
    {
        return make_uchar3(
                static_cast<unsigned char>(value.x),
                static_cast<unsigned char>(value.y),
                static_cast<unsigned char>(value.z)
                );
    }
};

template <tex_read_mode ReadMode>
struct map_texel_type<vector<4, unorm<8>>, ReadMode>
{
    using device_type = uchar4;
    using host_type   = vector<4, unorm<8>>;

    VSNRAY_FUNC static uchar4 cast(vector<4, unorm<8>> const& value)
    {
        return make_uchar4(
                static_cast<unsigned char>(value.x),
                static_cast<unsigned char>(value.y),
                static_cast<unsigned char>(value.z),
                static_cast<unsigned char>(value.w)
                );
    }
};

template <tex_read_mode ReadMode>
struct map_texel_type<vector<4, float>, ReadMode>
{
    using device_type = float4;
    using host_type   = vector<4, float>;

    VSNRAY_FUNC static float4 cast(vector<4, float> const& value)
    {
        return make_float4( value.x, value.y, value.z, value.w );
    }
};

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_UTIL_H
