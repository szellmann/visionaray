// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_UTIL_H
#define VSNRAY_CUDA_UTIL_H

#include <cuda_runtime_api.h>

#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>

namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
// cuda texel type <-> visionaray texel type
//

template <typename Type>
struct map_texel_type
{
    using device_type = Type;
    using host_type   = Type;
};


//-------------------------------------------------------------------------------------------------
// cuda -> visionaray
//

template <>
struct map_texel_type<uchar4>
{
    using device_type = uchar4;
    using host_type   = vector<4, unorm<8>>;

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
struct map_texel_type<float4>
{
    using device_type = float4;
    using host_typ    = vector<4, float>;

    VSNRAY_FUNC static vector<4, float> cast(float4 const& value)
    {
        return vector<4, float>( value.x, value.y, value.z, value.w );
    }
};


//-------------------------------------------------------------------------------------------------
// cuda <- visionaray
//

template <>
struct map_texel_type<vector<4, unorm<8>>>
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

template <>
struct map_texel_type<vector<4, float>>
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
