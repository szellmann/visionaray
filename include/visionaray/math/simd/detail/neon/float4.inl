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

VSNRAY_FORCE_INLINE float4::basic_float(float x, float y, float z, float w)
{
    // TODO: check if this can be done in the initializer list
    VSNRAY_ALIGN(16) float data[4] = { x, y, z, w };
    value = vld1q_f32(data);
}

VSNRAY_FORCE_INLINE float4::basic_float(float const v[4])
    : value(vld1q_f32(v))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(float s)
    : value(vdupq_n_f32(s))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(int32x4_t const& i)
    : value(vcvtq_f32_s32(i))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(float32x4_t const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE float4::operator float32x4_t() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE int4 reinterpret_as_int(float4 const& a)
{
    return vreinterpretq_s32_f32(a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int4 convert_to_int(float4 const& a)
{
    return vcvtq_s32_f32(a);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE float4 select(mask4 const& m, float4 const& a, float4 const& b)
{
    return vbslq_f32(m.i, a, b);
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

VSNRAY_FORCE_INLINE float4 load(float const src[4])
{
    return vld1q_f32(src);
}

VSNRAY_FORCE_INLINE void store(float dst[4], float4 const& v)
{
    vst1q_f32(dst, v);
}

template <unsigned I>
VSNRAY_FORCE_INLINE float& get(float4& v)
{
    static_assert(I < 4, "Index out of range for SIMD vector access");

    return reinterpret_cast<float*>(&v)[I];
}

template <unsigned I>
VSNRAY_FORCE_INLINE float const& get(float4 const& v)
{
    static_assert(I < 4, "Index out of range for SIMD vector access");

    return reinterpret_cast<float const*>(&v)[I];
}


//-------------------------------------------------------------------------------------------------
// Transposition
//

template <int U0, int U1, int V2, int V3>
VSNRAY_FORCE_INLINE float4 shuffle(float4 const& u, float4 const& v)
{
    // TODO: are dedicated implementations using
    // vget_{low|high}, vcombine, vzip, etc. faster?
    auto uu = reinterpret_cast<float const*>(&u);
    auto vv = reinterpret_cast<float const*>(&v);

    return float4(uu[U0], uu[U1], vv[V2], vv[V3]);
}

template <int V0, int V1, int V2, int V3>
VSNRAY_FORCE_INLINE float4 shuffle(float4 const& v)
{
    // TODO: are dedicated implementations using
    // vget_{low|high}, vcombine, vzip, etc. faster?
    auto vv = reinterpret_cast<float const*>(&v);

    return float4(vv[V0], vv[V1], vv[V2], vv[V3]);
}

VSNRAY_FORCE_INLINE float4 move_lo(float4 const& u, float4 const& v)
{
    // r1 := u1, r2 := u2, r3 := v1, r4 := v2
    auto ulo = vget_low_f32(u);
    auto vlo = vget_low_f32(v);
    return vcombine_f32(ulo, vlo);
}

VSNRAY_FORCE_INLINE float4 move_hi(float4 const& u, float4 const& v)
{
    // r1 := v3, r2 := v4, r3 := u3, r4 := u4
    auto uhi = vget_high_f32(u);
    auto vhi = vget_high_f32(v);
    return vcombine_f32(vhi, uhi);
}

VSNRAY_FORCE_INLINE float4 interleave_lo(float4 const& u, float4 const& v)
{
    // r1 := u1, r2 := v1, r3 := u2, r4 := v2
    auto ulo = vget_low_f32(u);
    auto vlo = vget_low_f32(v);
    auto zip = vzip_f32(ulo, vlo);
    return vcombine_f32(zip.val[0], zip.val[1]);
}

VSNRAY_FORCE_INLINE float4 interleave_hi(float4 const& u, float4 const& v)
{
    // r1 := u3, r2 := v3, r3 := u4, r4 := v4
    auto uhi = vget_high_f32(u);
    auto vhi = vget_high_f32(v);
    auto zip = vzip_f32(uhi, vhi);
    return vcombine_f32(zip.val[0], zip.val[1]);
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetics
//

VSNRAY_FORCE_INLINE float4 operator+(float4 const& v)
{
    return vaddq_f32(vdupq_n_f32(0.0f), v);
}

VSNRAY_FORCE_INLINE float4 operator-(float4 const& v)
{
    return vnegq_f32(v);
}

VSNRAY_FORCE_INLINE float4 operator+(float4 const& u, float4 const& v)
{
    return vaddq_f32(u, v);
}

VSNRAY_FORCE_INLINE float4 operator-(float4 const& u, float4 const& v)
{
    return vsubq_f32(u, v);
}

VSNRAY_FORCE_INLINE float4 operator*(float4 const& u, float4 const& v)
{
    return vmulq_f32(u, v);
}

VSNRAY_FORCE_INLINE float4 operator/(float4 const& u, float4 const& v)
{
    // NEON has no division instruction, so multiply by reciprocal
    // and refine with a couple of Newton-Raphson steps
    float32x4_t rcp = vrecpeq_f32(v);
    // vrecpsq_f32 conveniently performs a Newton-Raphson step
    rcp = vmulq_f32(vrecpsq_f32(v, rcp), rcp);
    rcp = vmulq_f32(vrecpsq_f32(v, rcp), rcp);
    return vmulq_f32(u, rcp);
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE float4 operator&(float4 const& u, float4 const& v)
{
    int32x4_t ui = vreinterpretq_s32_f32(u);
    int32x4_t vi = vreinterpretq_s32_f32(v);
    int32x4_t ri = vandq_s32(ui, vi);
    return vreinterpretq_f32_s32(ri);
}

VSNRAY_FORCE_INLINE float4 operator|(float4 const& u, float4 const& v)
{
    int32x4_t ui = vreinterpretq_s32_f32(u);
    int32x4_t vi = vreinterpretq_s32_f32(v);
    int32x4_t ri = vorrq_s32(ui, vi);
    return vreinterpretq_f32_s32(ri);
}

VSNRAY_FORCE_INLINE float4 operator^(float4 const& u, float4 const& v)
{
    int32x4_t ui = vreinterpretq_s32_f32(u);
    int32x4_t vi = vreinterpretq_s32_f32(v);
    int32x4_t ri = veorq_s32(ui, vi);
    return vreinterpretq_f32_s32(ri);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE float4 operator&&(float4 const& u, float4 const& v)
{
    int32x4_t ui = vreinterpretq_s32_f32(u);
    int32x4_t vi = vreinterpretq_s32_f32(v);
    int32x4_t ri = vandq_s32(ui, vi);
    return vreinterpretq_f32_s32(ri);
}

VSNRAY_FORCE_INLINE float4 operator||(float4 const& u, float4 const& v)
{
    int32x4_t ui = vreinterpretq_s32_f32(u);
    int32x4_t vi = vreinterpretq_s32_f32(v);
    int32x4_t ri = vorrq_s32(ui, vi);
    return vreinterpretq_f32_s32(ri);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator<(float4 const& u, float4 const& v)
{
    return vcltq_f32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>(float4 const& u, float4 const& v)
{
    return vcgtq_f32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator<=(float4 const& u, float4 const& v)
{
    return vcleq_f32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>=(float4 const& u, float4 const& v)
{
    return vcgeq_f32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator==(float4 const& u, float4 const& v)
{
    return vceqq_f32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator!=(float4 const& u, float4 const& v)
{
    return vmvnq_u32(vceqq_f32(u, v));
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE float4 min(float4 const& u, float4 const& v)
{
    return vminq_f32(u, v);
}

VSNRAY_FORCE_INLINE float4 max(float4 const& u, float4 const& v)
{
    return vmaxq_f32(u, v);
}

VSNRAY_FORCE_INLINE float4 saturate(float4 const& u)
{
    return vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(u, vdupq_n_f32(1.0f)));
}

VSNRAY_FORCE_INLINE float4 abs(float4 const& u)
{
    int4 a = vreinterpretq_s32_f32(u);
    int4 b = vdupq_n_s32(0x7FFFFFFF);
    return vreinterpretq_f32_s32(vandq_s32(a, b));
}

VSNRAY_FORCE_INLINE float4 round(float4 const& v)
{
    // Mask out the signbits of v
    int32x4_t s = vandq_s32(vreinterpretq_s32_f32(v), vdupq_n_s32(0x80000000));
    // Magic number: 2^23 with the signbits of v
    int32x4_t m = vorrq_s32(s, vdupq_n_s32(0x4B000000));
    float32x4_t x = vaddq_f32(v, vreinterpretq_f32_s32(m));
    float32x4_t y = vsubq_f32(x, vreinterpretq_f32_s32(m));

    return y;
}

VSNRAY_FORCE_INLINE float4 ceil(float4 const& v)
{
    // i = trunc(v)
    float32x4_t i = vcvtq_f32_s32(vcvtq_s32_f32(v));
    // r = i < v ? i i + 1 : i
    uint32x4_t t = vcltq_f32(i, v);
    int32x4_t st = vreinterpretq_s32_u32(t); // underflow: 0xFFFFFFFF -> -1
    float32x4_t d = vcvtq_f32_s32(st); // signed mask to float (0.0f or -1.0f)
    float32x4_t r = vsubq_f32(i, d);

    return r;
}

VSNRAY_FORCE_INLINE float4 floor(float4 const& v)
{
    // i = trunc(v)
    float32x4_t i = vcvtq_f32_s32(vcvtq_s32_f32(v));
    // r = i > v ? i - 1 : i
    uint32x4_t t = vcgtq_f32(i, v);
    int32x4_t st = vreinterpretq_s32_u32(t); // underflow: 0xFFFFFFFF -> -1
    float32x4_t d = vcvtq_f32_s32(st); // signed mask to float (0.0f or -1.0f)
    float32x4_t r = vaddq_f32(i, d);

    return r;
}

VSNRAY_FORCE_INLINE float4 sqrt(float4 const& v)
{
    float32x4_t rsqrt = vrsqrteq_f32(v);
    // Newton-Raphson step
    rsqrt = vrsqrtsq_f32(float32x4_t(v) * rsqrt, rsqrt) * rsqrt;
    // rsqrt => sqrt
    return float32x4_t(v) * rsqrt;
}

VSNRAY_FORCE_INLINE mask4 isinf(float4 const& v)
{
    VSNRAY_ALIGN(16) float values[4] = {};
    store(values, v);

    return mask4(
            std::isinf(values[0]),
            std::isinf(values[1]),
            std::isinf(values[2]),
            std::isinf(values[3])
            );
}

VSNRAY_FORCE_INLINE mask4 isnan(float4 const& v)
{
    return v != v;
}

VSNRAY_FORCE_INLINE mask4 isfinite(float4 const& v)
{
    return !(isinf(v) | isnan(v));
}

} // simd
} // MATH_NAMESPACE
