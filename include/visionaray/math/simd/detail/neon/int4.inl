// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// int4 members
//

VSNRAY_FORCE_INLINE int4::basic_int(int x, int y, int z, int w)
{
    // TODO: check if this can be done in the initializer list
    VSNRAY_ALIGN(16) int data[4] = { x, y, z, w };
    value = vld1q_s32(data);
}

VSNRAY_FORCE_INLINE int4::basic_int(int const v[4])
    : value(vld1q_s32(v))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(int s)
    : value(vdupq_n_s32(s))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(unsigned s)
    : value(vdupq_n_s32(static_cast<int>(s)))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(basic_float<float32x4_t> const& f)
    : value(vcvtq_s32_f32(f))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(int32x4_t const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE int4::operator int32x4_t() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE float4 reinterpret_as_float(int4 const& a)
{
    return vreinterpretq_f32_s32(a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE float4 convert_to_float(int4 const& a)
{
    return vcvtq_f32_s32(a);
}


//-------------------------------------------------------------------------------------------------
// Sign-extend
//

VSNRAY_FORCE_INLINE void sign_extend(int4& dst, char const* a4)
{
    int4 a;
    memcpy(&a, a4, 4 * sizeof(char));
    // _mm_cvtepi8_epi32 from SSE2Neon:
    int8x16_t s8x16 = vreinterpretq_s8_s64(*(int64x2_t*)&a);
    int16x8_t s16x8 = vmovl_s8(vget_low_s8(s8x16));
    int32x4_t s32x4 = vmovl_s16(vget_low_s16(s16x8));
    dst = (int32x4_t)vreinterpretq_s64_s32(s32x4);
}

VSNRAY_FORCE_INLINE void sign_extend(int4& dst, unsigned char const* a4)
{
    int4 a;
    memcpy(&a, a4, 4 * sizeof(unsigned char));
    // _mm_cvtepu8_epi32 from SSE2Neon:
    uint8x16_t u8x16 = vreinterpretq_u8_s32(a);
    uint16x8_t u16x8 = vmovl_u8(vget_low_u8(u8x16));
    uint32x4_t u32x4 = vmovl_u16(vget_low_u16(u16x8));
    dst = vreinterpretq_s32_u32(u32x4);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE int4 select(mask4 const& m, int4 const& a, int4 const& b)
{
    return vbslq_s32(m.i, a, b);
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

VSNRAY_FORCE_INLINE void store(int dst[4], int4 const& v)
{
    vst1q_s32(dst, v);
}

VSNRAY_FORCE_INLINE void store(unsigned dst[4], uint32x4_t const& v)
{
    vst1q_u32(dst, v);
}

VSNRAY_FORCE_INLINE void store(int dst[4], uint32x4_t const& v)
{
    vst1q_u32(reinterpret_cast<unsigned*>(dst), v);
}

VSNRAY_FORCE_INLINE void store(unsigned dst[4], int4 const& v)
{
    vst1q_s32(reinterpret_cast<int*>(dst), v);
}

template <unsigned I>
VSNRAY_FORCE_INLINE int& get(int4& v)
{
    static_assert(I < 4, "Index out of range for SIMD vector access");

    return reinterpret_cast<int*>(&v)[I];
}

template <unsigned I>
VSNRAY_FORCE_INLINE int const& get(int4 const& v)
{
    static_assert(I < 4, "Index out of range for SIMD vector access");

    return reinterpret_cast<int const*>(&v)[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmethic
//

VSNRAY_FORCE_INLINE int4 operator+(int4 const& v)
{
    return vaddq_s32(vdupq_n_s32(0), v);
}

VSNRAY_FORCE_INLINE int4 operator-(int4 const& v)
{
    return vnegq_s32(v);
}

VSNRAY_FORCE_INLINE int4 operator+(int4 const& u, int4 const& v)
{
    return vaddq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator-(int4 const& u, int4 const& v)
{
    return vsubq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator*(int4 const& u, int4 const& v)
{
    return vmulq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator/(int4 const& u, int4 const& v)
{
    return convert_to_int( convert_to_float(u) / convert_to_float(v) );
}

VSNRAY_FORCE_INLINE int4 operator%(int4 const& u, int4 const& v)
{
    float4 uf = convert_to_float(u);
    float4 vf = convert_to_float(v);

    int4   t0 = u / v;
    float4 t1 = convert_to_float(t0);
    float4 t2 = t1 * vf;
    float4 t3 = uf - t2;

    return convert_to_int(t3);
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE int4 operator&(int4 const& u, int4 const& v)
{
    return vandq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator|(int4 const& u, int4 const& v)
{
    return vorrq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator^(int4 const& u, int4 const& v)
{
    return veorq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator<<(int4 const& a, int count)
{
    return vshlq_s32(a, vdupq_n_s32(count));
}

VSNRAY_FORCE_INLINE int4 operator>>(int4 const& a, int count)
{
    return vshlq_s32(a, vdupq_n_s32(-count));
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE int4 operator&&(int4 const& u, int4 const& v)
{
    return vandq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator||(int4 const& u, int4 const& v)
{
    return vorrq_s32(u, v);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator<(int4 const& u, int4 const& v)
{
    return vcltq_s32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>(int4 const& u, int4 const& v)
{
    return vcgtq_s32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator<=(int4 const& u, int4 const& v)
{
    return vcleq_s32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>=(int4 const& u, int4 const& v)
{
    return vcgeq_s32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator==(int4 const& u, int4 const& v)
{
    return vceqq_s32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator!=(int4 const& u, int4 const& v)
{
    return vmvnq_u32(vceqq_s32(u, v));
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE int4 min(int4 const& u, int4 const& v)
{
    return vminq_s32(u, v);
}

VSNRAY_FORCE_INLINE int4 max(int4 const& u, int4 const& v)
{
    return vmaxq_s32(u, v);
}

} // simd
} // MATH_NAMESPACE
