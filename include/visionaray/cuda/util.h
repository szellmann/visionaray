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
        return vector<4, unorm<8>>(
                value.x / 255.0f,
                value.y / 255.0f,
                value.z / 255.0f,
                value.w / 255.0f
                );
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
struct map_texel_type<vector<4, unorm<8>>, ReadMode>
{
    using device_type = uchar4;
    using host_type   = vector<4, unorm<8>>;

    VSNRAY_FUNC static uchar4 cast(vector<4, unorm<8>> const& value)
    {
        return make_uchar4(
                value.x * 255,
                value.y * 255,
                value.z * 255,
                value.w * 255
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
