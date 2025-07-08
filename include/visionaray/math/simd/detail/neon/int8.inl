// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// int8 members
//

VSNRAY_FORCE_INLINE int8::basic_int(int32x4_t const& v1, int32x4_t const& v2)
{
    value[0] = v1;
    value[1] = v2;
}

VSNRAY_FORCE_INLINE int8::basic_int(
        int x1,
        int x2,
        int x3,
        int x4,
        int x5,
        int x6,
        int x7,
        int x8
        )
{
    VSNRAY_ALIGN(16) int data[8] = { x1, x2, x3, x4, x5, x6, x7, x8 };
    value[0] = vld1q_s32(&data[0]);
    value[1] = vld1q_s32(&data[4]);
}

VSNRAY_FORCE_INLINE int8::basic_int(int const v[8])
{
    value[0] = vld1q_s32(&v[0]);
    value[1] = vld1q_s32(&v[4]);
}

VSNRAY_FORCE_INLINE int8::basic_int(int s)
{
    value[0] = vdupq_n_s32(s);
    value[1] = vdupq_n_s32(s);
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE float8 reinterpret_as_float(int8 const& a)
{
    return float8(
        vreinterpretq_f32_s32(a.value[0]),
        vreinterpretq_f32_s32(a.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE float8 convert_to_float(int8 const& a)
{
    return float8(vcvtq_f32_s32(a.value[0]), vcvtq_f32_s32(a.value[1]));
}


//-------------------------------------------------------------------------------------------------
// Sign-extend
//

VSNRAY_FORCE_INLINE void sign_extend(int8& dst, char const* a8)
{
    sign_extend((int4&)dst.value[0], &a8[0]);
    sign_extend((int4&)dst.value[1], &a8[4]);
}

VSNRAY_FORCE_INLINE void sign_extend(int8& dst, unsigned char const* a8)
{
    sign_extend((int4&)dst.value[0], &a8[0]);
    sign_extend((int4&)dst.value[1], &a8[4]);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE int8 select(mask8 const& m, int8 const& a, int8 const& b)
{
    return int8(
        vbslq_s32(m.i[0], a.value[0], b.value[0]),
        vbslq_s32(m.i[1], a.value[1], b.value[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

VSNRAY_FORCE_INLINE void store(int dst[8], int8 const& v)
{
    vst1q_s32(&dst[0], v.value[0]);
    vst1q_s32(&dst[4], v.value[1]);
}


//-------------------------------------------------------------------------------------------------
// Basic arithmethic
//

VSNRAY_FORCE_INLINE int8 operator+(int8 const& v)
{
    return int8(
        vaddq_s32(vdupq_n_s32(0), v.value[0]),
        vaddq_s32(vdupq_n_s32(0), v.value[1])
        );
}

VSNRAY_FORCE_INLINE int8 operator-(int8 const& v)
{
    return int8(vnegq_s32(v.value[0]), vnegq_s32(v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator+(int8 const& u, int8 const& v)
{
    return int8(vaddq_s32(u.value[0], v.value[0]), vaddq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator-(int8 const& u, int8 const& v)
{
    return int8(vsubq_s32(u.value[0], v.value[0]), vsubq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator*(int8 const& u, int8 const& v)
{
    return int8(vmulq_s32(u.value[0], v.value[0]), vmulq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator/(int8 const& u, int8 const& v)
{
    return convert_to_int( convert_to_float(u) / convert_to_float(v) );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE int8 operator&(int8 const& u, int8 const& v)
{
    return int8(vandq_s32(u.value[0], v.value[0]), vandq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator|(int8 const& u, int8 const& v)
{
    return int8(vorrq_s32(u.value[0], v.value[0]), vorrq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator^(int8 const& u, int8 const& v)
{
    return int8(veorq_s32(u.value[0], v.value[0]), veorq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator<<(int8 const& a, int count)
{
    return int8(
        vshlq_s32(a.value[0], vdupq_n_s32(count)),
        vshlq_s32(a.value[1], vdupq_n_s32(count))
        );
}

VSNRAY_FORCE_INLINE int8 operator>>(int8 const& a, int count)
{
    return int8(
        vshlq_s32(a.value[0], vdupq_n_s32(-count)),
        vshlq_s32(a.value[1], vdupq_n_s32(-count))
        );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE int8 operator&&(int8 const& u, int8 const& v)
{
    return int8(vandq_s32(u.value[0], v.value[0]), vandq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 operator||(int8 const& u, int8 const& v)
{
    return int8(vorrq_s32(u.value[0], v.value[0]), vorrq_s32(u.value[1], v.value[1]));
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask8 operator<(int8 const& u, int8 const& v)
{
    return mask8(vcltq_s32(u.value[0], v.value[0]), vcltq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator>(int8 const& u, int8 const& v)
{
    return mask8(vcgtq_s32(u.value[0], v.value[0]), vcgtq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator<=(int8 const& u, int8 const& v)
{
    return mask8(vcleq_s32(u.value[0], v.value[0]), vcleq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator>=(int8 const& u, int8 const& v)
{
    return mask8(vcgeq_s32(u.value[0], v.value[0]), vcgeq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator==(int8 const& u, int8 const& v)
{
    return mask8(vceqq_s32(u.value[0], v.value[0]), vceqq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator!=(int8 const& u, int8 const& v)
{
    return mask8(
        vmvnq_u32(vceqq_s32(u.value[0], v.value[0])),
        vmvnq_u32(vceqq_s32(u.value[1], v.value[1]))
        );
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE int8 min(int8 const& u, int8 const& v)
{
    return int8(vminq_s32(u.value[0], v.value[0]), vminq_s32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE int8 max(int8 const& u, int8 const& v)
{
    return int8(vmaxq_s32(u.value[0], v.value[0]), vmaxq_s32(u.value[1], v.value[1]));
}

} // simd
} // MATH_NAMESPACE
