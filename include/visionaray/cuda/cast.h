// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_CAST_H
#define VSNRAY_CUDA_CAST_H 1

#include <cuda_runtime.h>
#include <vector_functions.h>

#include <visionaray/detail/macros.h>
#include <visionaray/math/norm.h>
#include <visionaray/math/vector.h>

namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
// cuda::cast
//
// Cast between Visionaray and CUDA vector types
//
//-------------------------------------------------------------------------------------------------

namespace detail
{

//-------------------------------------------------------------------------------------------------
// cuda <-> visionaray (general types)
//

template <typename Dest, typename Source>
VSNRAY_FUNC
inline Dest cast_impl(Dest /* */, Source const& value)
{
    return static_cast<Dest>(value);
}


//-------------------------------------------------------------------------------------------------
// cuda -> visionaray
//

// vec2 ---------------------------------------------------

VSNRAY_FUNC
inline vector<2, char> cast_impl(vector<2, char> /* */, char2 const& value)
{
    return vector<2, char>(value.x, value.y);
}

VSNRAY_FUNC
inline vector<2, unsigned char> cast_impl(vector<2, unsigned char> /* */, uchar2 const& value)
{
    return vector<2, unsigned char>(value.x, value.y);
}

VSNRAY_FUNC
inline vector<2, short> cast_impl(vector<2, short> /* */, short2 const& value)
{
    return vector<2, short>(value.x, value.y);
}

VSNRAY_FUNC
inline vector<2, unsigned short> cast_impl(vector<2, unsigned short> /* */, ushort2 const& value)
{
    return vector<2, unsigned short>(value.x, value.y);
}

VSNRAY_FUNC
inline vector<2, int> cast_impl(vector<2, int> /* */, int2 const& value)
{
    return vector<2, int>(value.x, value.y);
}

VSNRAY_FUNC
inline vector<2, unsigned int> cast_impl(vector<2, unsigned int> /* */, uint2 const& value)
{
    return vector<2, unsigned int>(value.x, value.y);
}

VSNRAY_FUNC
inline vector<2, float> cast_impl(vector<2, float> /* */, float2 const& value)
{
    return vector<2, float>(value.x, value.y);
}

// vec3 ---------------------------------------------------

VSNRAY_FUNC
inline vector<3, char> cast_impl(vector<3, char> /* */, char3 const& value)
{
    return vector<3, char>(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline vector<3, unsigned char> cast_impl(vector<3, unsigned char> /* */, uchar3 const& value)
{
    return vector<3, unsigned char>(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline vector<3, short> cast_impl(vector<3, short> /* */, short3 const& value)
{
    return vector<3, short>(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline vector<3, unsigned short> cast_impl(vector<3, unsigned short> /* */, ushort3 const& value)
{
    return vector<3, unsigned short>(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline vector<3, int> cast_impl(vector<3, int> /* */, int3 const& value)
{
    return vector<3, int>(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline vector<3, unsigned int> cast_impl(vector<3, unsigned int> /* */, uint3 const& value)
{
    return vector<3, unsigned int>(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline vector<3, float> cast_impl(vector<3, float> /* */, float3 const& value)
{
    return vector<3, float>(value.x, value.y, value.z);
}

// vec4 ---------------------------------------------------

VSNRAY_FUNC
inline vector<4, char> cast_impl(vector<4, char> /* */, char4 const& value)
{
    return vector<4, char>(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline vector<4, unsigned char> cast_impl(vector<4, unsigned char> /* */, uchar4 const& value)
{
    return vector<4, unsigned char>(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline vector<4, short> cast_impl(vector<4, short> /* */, short4 const& value)
{
    return vector<4, short>(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline vector<4, unsigned short> cast_impl(vector<4, unsigned short> /* */, ushort4 const& value)
{
    return vector<4, unsigned short>(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline vector<4, int> cast_impl(vector<4, int> /* */, int4 const& value)
{
    return vector<4, int>(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline vector<4, unsigned int> cast_impl(vector<4, unsigned int> /* */, uint4 const& value)
{
    return vector<4, unsigned int>(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline vector<4, float> cast_impl(vector<4, float> /* */, float4 const& value)
{
    return vector<4, float>(value.x, value.y, value.z, value.w);
}

//-------------------------------------------------------------------------------------------------
// cuda <- visionaray
//

// vec2 ---------------------------------------------------

VSNRAY_FUNC
inline char2 cast_impl(char2 /* */, vector<2, char> const& value)
{
    return make_char2(value.x, value.y);
}

VSNRAY_FUNC
inline uchar2 cast_impl(uchar2 /* */, vector<2, unsigned char> const& value)
{
    return make_uchar2(value.x, value.y);
}

VSNRAY_FUNC
inline uchar2 cast_impl(uchar2 /* */, vector<2, unorm<8>> const& value)
{
    return make_uchar2(
            static_cast<unsigned char>(value.x),
            static_cast<unsigned char>(value.y)
            );
}

VSNRAY_FUNC
inline short2 cast_impl(short2 /* */, vector<2, short> const& value)
{
    return make_short2(value.x, value.y);
}

VSNRAY_FUNC
inline ushort2 cast_impl(ushort2 /* */, vector<2, unsigned short> const& value)
{
    return make_ushort2(value.x, value.y);
}

VSNRAY_FUNC
inline ushort2 cast_impl(ushort2 /* */, vector<2, unorm<16>> const& value)
{
    return make_ushort2(
            static_cast<unsigned short>(value.x),
            static_cast<unsigned short>(value.y)
            );
}

VSNRAY_FUNC
inline int2 cast_impl(int2 /* */, vector<2, int> const& value)
{
    return make_int2(value.x, value.y);
}

VSNRAY_FUNC
inline uint2 cast_impl(uint2 /* */, vector<2, unsigned int> const& value)
{
    return make_uint2(value.x, value.y);
}

VSNRAY_FUNC
inline uint2 cast_impl(uint2 /* */, vector<2, unorm<32>> const& value)
{
    return make_uint2(
            static_cast<unsigned int>(value.x),
            static_cast<unsigned int>(value.y)
            );
}

VSNRAY_FUNC
inline float2 cast_impl(float2 /* */, vector<2, float> const& value)
{
    return make_float2(value.x, value.y);
}

// vec3 ---------------------------------------------------

VSNRAY_FUNC
inline char3 cast_impl(char3 /* */, vector<3, char> const& value)
{
    return make_char3(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline uchar3 cast_impl(uchar3 /* */, vector<3, unsigned char> const& value)
{
    return make_uchar3(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline uchar3 cast_impl(uchar3 /* */, vector<3, unorm<8>> const& value)
{
    return make_uchar3(
            static_cast<unsigned char>(value.x),
            static_cast<unsigned char>(value.y),
            static_cast<unsigned char>(value.z)
            );
}

VSNRAY_FUNC
inline short3 cast_impl(short3 /* */, vector<3, short> const& value)
{
    return make_short3(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline ushort3 cast_impl(ushort3 /* */, vector<3, unsigned short> const& value)
{
    return make_ushort3(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline ushort3 cast_impl(ushort3 /* */, vector<3, unorm<16>> const& value)
{
    return make_ushort3(
            static_cast<unsigned short>(value.x),
            static_cast<unsigned short>(value.y),
            static_cast<unsigned short>(value.z)
            );
}

VSNRAY_FUNC
inline int3 cast_impl(int3 /* */, vector<3, int> const& value)
{
    return make_int3(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline uint3 cast_impl(uint3 /* */, vector<3, unsigned int> const& value)
{
    return make_uint3(value.x, value.y, value.z);
}

VSNRAY_FUNC
inline uint3 cast_impl(uint3 /* */, vector<3, unorm<32>> const& value)
{
    return make_uint3(
            static_cast<unsigned int>(value.x),
            static_cast<unsigned int>(value.y),
            static_cast<unsigned int>(value.z)
            );
}

VSNRAY_FUNC
inline float3 cast_impl(float3 /* */, vector<3, float> const& value)
{
    return make_float3(value.x, value.y, value.z);
}

// vec4 ---------------------------------------------------

VSNRAY_FUNC
inline char4 cast_impl(char4 /* */, vector<4, char> const& value)
{
    return make_char4(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline uchar4 cast_impl(uchar4 /* */, vector<4, unsigned char> const& value)
{
    return make_uchar4(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline uchar4 cast_impl(uchar4 /* */, vector<4, unorm<8>> const& value)
{
    return make_uchar4(
            static_cast<unsigned char>(value.x),
            static_cast<unsigned char>(value.y),
            static_cast<unsigned char>(value.z),
            static_cast<unsigned char>(value.w)
            );
}

VSNRAY_FUNC
inline short4 cast_impl(short4 /* */, vector<4, short> const& value)
{
    return make_short4(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline ushort4 cast_impl(ushort4 /* */, vector<4, unsigned short> const& value)
{
    return make_ushort4(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline ushort4 cast_impl(ushort4 /* */, vector<4, unorm<16>> const& value)
{
    return make_ushort4(
            static_cast<unsigned short>(value.x),
            static_cast<unsigned short>(value.y),
            static_cast<unsigned short>(value.z),
            static_cast<unsigned short>(value.w)
            );
}

VSNRAY_FUNC
inline int4 cast_impl(int4 /* */, vector<4, int> const& value)
{
    return make_int4(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline uint4 cast_impl(uint4 /* */, vector<4, unsigned int> const& value)
{
    return make_uint4(value.x, value.y, value.z, value.w);
}

VSNRAY_FUNC
inline uint4 cast_impl(uint4 /* */, vector<4, unorm<32>> const& value)
{
    return make_uint4(
            static_cast<unsigned int>(value.x),
            static_cast<unsigned int>(value.y),
            static_cast<unsigned int>(value.z),
            static_cast<unsigned int>(value.w)
            );
}

VSNRAY_FUNC
inline float4 cast_impl(float4 /* */, vector<4, float> const& value)
{
    return make_float4(value.x, value.y, value.z, value.w);
}

} // detail


//-------------------------------------------------------------------------------------------------
// Interface
//

template <typename Dest, typename Source>
VSNRAY_FUNC
inline Dest cast(Source const& value)
{
    return detail::cast_impl(Dest{}, value);
}

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_CAST_H
