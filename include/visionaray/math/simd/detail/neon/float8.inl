// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cmath>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float4 members
//

VSNRAY_FORCE_INLINE float8::basic_float(float32x4_t const& v1, float32x4_t const& v2)
{
    value[0] = v1;
    value[1] = v2;
}

VSNRAY_FORCE_INLINE float8::basic_float(
        float x1,
        float x2,
        float x3,
        float x4,
        float x5,
        float x6,
        float x7,
        float x8
        )
{
    VSNRAY_ALIGN(16) float data[8] = { x1, x2, x3, x4, x5, x6, x7, x8 };
    value[0] = vld1q_f32(&data[0]);
    value[1] = vld1q_f32(&data[4]);
}

VSNRAY_FORCE_INLINE float8::basic_float(float const v[8])
{
    value[0] = vld1q_f32(&v[0]);
    value[1] = vld1q_f32(&v[4]);
}

VSNRAY_FORCE_INLINE float8::basic_float(float s)
{
    value[0] = vdupq_n_f32(s);
    value[1] = vdupq_n_f32(s);
}

VSNRAY_FORCE_INLINE float8::basic_float(int8 const& i)
{
    value[0] = vcvtq_f32_s32(i.value[0]);
    value[1] = vcvtq_f32_s32(i.value[1]);
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE int8 reinterpret_as_int(float8 const& a)
{
    return int8(
        vreinterpretq_s32_f32(a.value[0]),
        vreinterpretq_s32_f32(a.value[1])
        );
}

VSNRAY_FORCE_INLINE uint8 reinterpret_as_uint(float8 const& a)
{
    return uint8(
        vreinterpretq_u32_f32(a.value[0]),
        vreinterpretq_u32_f32(a.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int8 convert_to_int(float8 const& a)
{
    return int8(vcvtq_s32_f32(a.value[0]), vcvtq_s32_f32(a.value[1]));
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE float8 select(mask8 const& m, float8 const& a, float8 const& b)
{
    return float8(
        vbslq_f32(m.i[0], a.value[0], b.value[0]),
        vbslq_f32(m.i[1], a.value[1], b.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

// VSNRAY_FORCE_INLINE float8 load(float const src[8])
// {
//     return float8(vld1q_f32(&src[0]), vld1q_f32(&src[4]));
// }

VSNRAY_FORCE_INLINE void store(float dst[8], float8 const& v)
{
    vst1q_f32(&dst[0], v.value[0]);
    vst1q_f32(&dst[4], v.value[1]);
}

template <unsigned I>
VSNRAY_FORCE_INLINE float& get(float8& v)
{
    static_assert(I < 8, "Index out of range for SIMD vector access");

    return reinterpret_cast<float*>(&v)[I];
}

template <unsigned I>
VSNRAY_FORCE_INLINE float const& get(float8 const& v)
{
    static_assert(I < 8, "Index out of range for SIMD vector access");

    return reinterpret_cast<float const*>(&v)[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetics
//

VSNRAY_FORCE_INLINE float8 operator+(float8 const& v)
{
    return float8(
        vaddq_f32(vdupq_n_f32(0.0f), v.value[0]),
        vaddq_f32(vdupq_n_f32(0.0f), v.value[1])
        );
}

VSNRAY_FORCE_INLINE float8 operator-(float8 const& v)
{
    return float8(vnegq_f32(v.value[0]), vnegq_f32(v.value[1]));
}

VSNRAY_FORCE_INLINE float8 operator+(float8 const& u, float8 const& v)
{
    return float8(vaddq_f32(u.value[0], v.value[0]), vaddq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE float8 operator-(float8 const& u, float8 const& v)
{
    return float8(vsubq_f32(u.value[0], v.value[0]), vsubq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE float8 operator*(float8 const& u, float8 const& v)
{
    return float8(vmulq_f32(u.value[0], v.value[0]), vmulq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE float8 operator/(float8 const& u, float8 const& v)
{
    // work around missing floating point div instruction (see float4 impl):
    return float8(
        float4(u.value[0]) / float4(v.value[0]),
        float4(u.value[1]) / float4(v.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE float8 operator&(float8 const& u, float8 const& v)
{
    return float8(
        float4(u.value[0]) & float4(v.value[0]),
        float4(u.value[1]) & float4(v.value[1])
        );
}

VSNRAY_FORCE_INLINE float8 operator|(float8 const& u, float8 const& v)
{
    return float8(
        float4(u.value[0]) | float4(v.value[0]),
        float4(u.value[1]) | float4(v.value[1])
        );
}

VSNRAY_FORCE_INLINE float8 operator^(float8 const& u, float8 const& v)
{
    return float8(
        float4(u.value[0]) ^ float4(v.value[0]),
        float4(u.value[1]) ^ float4(v.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE float8 operator&&(float8 const& u, float8 const& v)
{
    return float8(
        float4(u.value[0]) && float4(v.value[0]),
        float4(u.value[1]) && float4(v.value[1])
        );
}

VSNRAY_FORCE_INLINE float8 operator||(float8 const& u, float8 const& v)
{
    return float8(
        float4(u.value[0]) || float4(v.value[0]),
        float4(u.value[1]) || float4(v.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask8 operator<(float8 const& u, float8 const& v)
{
    return mask8(vcltq_f32(u.value[0], v.value[0]), vcltq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator>(float8 const& u, float8 const& v)
{
    return mask8(vcgtq_f32(u.value[0], v.value[0]), vcgtq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator<=(float8 const& u, float8 const& v)
{
    return mask8(vcleq_f32(u.value[0], v.value[0]), vcleq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator>=(float8 const& u, float8 const& v)
{
    return mask8(vcgeq_f32(u.value[0], v.value[0]), vcgeq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator==(float8 const& u, float8 const& v)
{
    return mask8(vceqq_f32(u.value[0], v.value[0]), vceqq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator!=(float8 const& u, float8 const& v)
{
    return mask8(
        vmvnq_u32(vceqq_f32(u.value[0], v.value[0])),
        vmvnq_u32(vceqq_f32(u.value[1], v.value[1]))
        );
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE float8 min(float8 const& u, float8 const& v)
{
    return float8(vminq_f32(u.value[0], v.value[0]), vminq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE float8 max(float8 const& u, float8 const& v)
{
    return float8(vmaxq_f32(u.value[0], v.value[0]), vmaxq_f32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE float8 saturate(float8 const& u)
{
    return float8(saturate(float4(u.value[0])), saturate(float4(u.value[1])));
}

VSNRAY_FORCE_INLINE float8 abs(float8 const& u)
{
    return float8(abs(float4(u.value[0])), abs(float4(u.value[1])));
}

VSNRAY_FORCE_INLINE float8 round(float8 const& u)
{
    return float8(round(float4(u.value[0])), round(float4(u.value[1])));
}

VSNRAY_FORCE_INLINE float8 ceil(float8 const& u)
{
    return float8(ceil(float4(u.value[0])), ceil(float4(u.value[1])));
}

VSNRAY_FORCE_INLINE float8 floor(float8 const& u)
{
    return float8(floor(float4(u.value[0])), floor(float4(u.value[1])));
}

VSNRAY_FORCE_INLINE float8 sqrt(float8 const& u)
{
    return float8(sqrt(float4(u.value[0])), sqrt(float4(u.value[1])));
}

} // simd
} // MATH_NAMESPACE
