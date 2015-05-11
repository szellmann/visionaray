// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_AVX_H
#define VSNRAY_MATH_SIMD_AVH_H

#include <visionaray/detail/macros.h>

#include "forward.h"
#include "intrinsics.h"

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

namespace MATH_NAMESPACE
{
namespace simd
{

template <>
class basic_float<__m256>
{
public:

    typedef __m256 value_type;
    __m256 value;

    VSNRAY_FORCE_INLINE basic_float() = default;

    VSNRAY_FORCE_INLINE basic_float
    (
        float x1, float x2, float x3, float x4,
        float x5, float x6, float x7, float x8
    )
        : value(_mm256_set_ps(x8, x7, x6, x5, x4, x3, x2, x1))
    {
    }

    VSNRAY_FORCE_INLINE basic_float(float const v[8])
        : value(_mm256_load_ps(v))
    {
    }

    VSNRAY_FORCE_INLINE basic_float(float s)
        : value(_mm256_set1_ps(s))
    {
    }

    VSNRAY_FORCE_INLINE basic_float(__m256i const& i)
        : value(_mm256_cvtepi32_ps(i))
    {
    }

    VSNRAY_FORCE_INLINE basic_float(__m256 const& v)
        : value(v)
    {
    }

    VSNRAY_FORCE_INLINE operator __m256() const
    {
        return value;
    }
};

template <>
class basic_int<__m256i>
{
public:

    typedef __m256i value_type;
    __m256i value;

    VSNRAY_FORCE_INLINE basic_int() = default;

    VSNRAY_FORCE_INLINE basic_int
    (
        int x1, int x2, int x3, int x4,
        int x5, int x6, int x7, int x8
    )
        : value(_mm256_set_epi32(x8, x7, x6, x5, x4, x3, x2, x1))
    {
    }

    VSNRAY_FORCE_INLINE basic_int(int const v[8])
        : value(_mm256_load_si256(reinterpret_cast<__m256i const*>(v)))
    {
    }

    VSNRAY_FORCE_INLINE basic_int(int s)
        : value(_mm256_set1_epi32(s))
    {
    }

    VSNRAY_FORCE_INLINE basic_int(unsigned s)
        : value(_mm256_set1_epi32(s))
    {
    }

    VSNRAY_FORCE_INLINE basic_int(basic_float<__m256> const& f)
        : value(_mm256_cvtps_epi32(f))
    {
    }

    VSNRAY_FORCE_INLINE basic_int(__m256i const& v)
        : value(v)
    {
    }

    VSNRAY_FORCE_INLINE operator __m256i() const
    {
        return value;
    }
};

template <>
union basic_mask<__m256, __m256i>
{
public:

    __m256  f;
    __m256i i;

    basic_mask() = default;

    VSNRAY_FORCE_INLINE basic_mask(__m256 m)
        : f(m)
    {
    }

    VSNRAY_FORCE_INLINE basic_mask(__m256i m)
        : i(m)
    {
    }

    VSNRAY_FORCE_INLINE basic_mask(bool b)
        : i( basic_int<__m256i>(b ? 0xFFFFFFFF : 0x00000000) )
    {
    }

    basic_mask(
            bool x1,
            bool x2,
            bool x3,
            bool x4,
            bool x5,
            bool x6,
            bool x7,
            bool x8
            )
        : i( basic_int<__m256i>(
                x1 ? 0xFFFFFFFF : 0x00000000,
                x2 ? 0xFFFFFFFF : 0x00000000,
                x3 ? 0xFFFFFFFF : 0x00000000,
                x4 ? 0xFFFFFFFF : 0x00000000,
                x5 ? 0xFFFFFFFF : 0x00000000,
                x6 ? 0xFFFFFFFF : 0x00000000,
                x7 ? 0xFFFFFFFF : 0x00000000,
                x8 ? 0xFFFFFFFF : 0x00000000
                ) )
    {
    }

    VSNRAY_FORCE_INLINE basic_mask(basic_float<__m256> const& m)
        : f(m)
    {
    }

    VSNRAY_FORCE_INLINE operator basic_float<__m256>() const
    {
        return f;
    }

};

typedef basic_float<__m256> float8;


inline float8 reinterpret_as_float(int8 const& a)
{
    return _mm256_castsi256_ps(a);
}

inline int8 reinterpret_as_int(float8 const& a)
{
    return _mm256_castps_si256(a);
}

VSNRAY_FORCE_INLINE float8 select(mask8 const& m, float8 const& a, float8 const& b)
{
    return _mm256_blendv_ps(b, a, m.f);
}

VSNRAY_FORCE_INLINE int8 select(mask8 const& m, int8 const& a, int8 const& b)
{
    return reinterpret_as_int( select(m, reinterpret_as_float(a), reinterpret_as_float(b)) );
}

//-------------------------------------------------------------------------------------------------
// float8
//

VSNRAY_FORCE_INLINE void store(float dst[8], float8 const& v)
{
    _mm256_store_ps(dst, v);
}


/* operators */

VSNRAY_FORCE_INLINE float8 operator-(float8 const& v)
{
    return _mm256_sub_ps(_mm256_setzero_ps(), v);
}

VSNRAY_FORCE_INLINE float8 operator+(float8 const& u, float8 const& v)
{
    return _mm256_add_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 operator-(float8 const& u, float8 const& v)
{
    return _mm256_sub_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 operator*(float8 const& u, float8 const& v)
{
    return _mm256_mul_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 operator/(float8 const& u, float8 const& v)
{
    return _mm256_div_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 operator&(float8 const& u, float8 const& v)
{
    return _mm256_and_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 operator|(float8 const& u, float8 const& v)
{
    return _mm256_or_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 operator^(float8 const& u, float8 const& v)
{
    return _mm256_xor_ps(u, v);
}

VSNRAY_FORCE_INLINE float8& operator+=(float8& u, float8 const& v)
{
    u = u + v;
    return u;
}

VSNRAY_FORCE_INLINE float8& operator-=(float8& u, float8 const& v)
{
    u = u - v;
    return u;
}

VSNRAY_FORCE_INLINE float8& operator*=(float8& u, float8 const& v)
{
    u = u * v;
    return u;
}

VSNRAY_FORCE_INLINE float8& operator/=(float8& u, float8 const& v)
{
    u = u / v;
    return u;
}

VSNRAY_FORCE_INLINE float8& operator&=(float8& u, float8 const& v)
{
    u = u & v;
    return u;
}

VSNRAY_FORCE_INLINE float8& operator|=(float8& u, float8 const& v)
{
    u = u | v;
    return u;
}

VSNRAY_FORCE_INLINE float8& operator^=(float8& u, float8 const& v)
{
    u = u ^ v;
    return u;
}



//-------------------------------------------------------------------------------------------------
// int8
//

VSNRAY_FORCE_INLINE void store(int dst[8], int8 const& v)
{
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
}

VSNRAY_FORCE_INLINE void store(unsigned dst[8], int8 const& v)
{
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
}

/* operators */

VSNRAY_FORCE_INLINE int8 operator+(int8 const& u, int8 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_add_epi32(u, v);
#else
    return int8(float8(u) + float8(v));
#endif
}

VSNRAY_FORCE_INLINE int8 operator-(int8 const& u, int8 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_sub_epi32(u, v);
#else
    return int8(float8(u) - float8(v));
#endif
}

VSNRAY_FORCE_INLINE int8 operator*(int8 const& u, int8 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_mul_epi32(u, v);
#else
    return int8(float8(u) * float8(v));
#endif
}

VSNRAY_FORCE_INLINE int8 operator/(int8 const& u, int8 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2 && VSNRAY_SIMD_HAS_SVML
    return _mm256_div_epi32(u, v);
#else
    return int8(float8(u) / float8(v));
#endif
}

VSNRAY_FORCE_INLINE int8 operator&(int8 const& u, int8 const& v)
{
    return reinterpret_as_int(reinterpret_as_float(u) & reinterpret_as_float(v));
}

VSNRAY_FORCE_INLINE int8 operator|(int8 const& u, int8 const& v)
{
    return reinterpret_as_int(reinterpret_as_float(u) | reinterpret_as_float(v));
}

VSNRAY_FORCE_INLINE int8 operator^(int8 const& u, int8 const& v)
{
    return reinterpret_as_int(reinterpret_as_float(u) ^ reinterpret_as_float(v));
}

VSNRAY_FORCE_INLINE int8 operator<<(int8 const& a, int count)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_slli_epi32(a, count);
#else
    __m128i lo = _mm256_castsi256_si128(a);
    __m128i hi = _mm256_extractf128_si256(a, 1);
            lo = _mm_slli_epi32(lo, count);
            hi = _mm_slli_epi32(hi, count);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
#endif
}

VSNRAY_FORCE_INLINE int8 operator>>(int8 const& a, int count)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_srai_epi32(a, count);
#else
    __m128i lo = _mm256_castsi256_si128(a);
    __m128i hi = _mm256_extractf128_si256(a, 1);
            lo = _mm_srai_epi32(lo, count);
            hi = _mm_srai_epi32(hi, count);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
#endif
}

VSNRAY_FORCE_INLINE int8& operator+=(int8& u, int8 const& v)
{
    u = u + v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator-=(int8& u, int8 const& v)
{
    u = u - v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator*=(int8& u, int8 const& v)
{
    u = u * v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator/=(int8& u, int8 const& v)
{
    u = u / v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator&=(int8& u, int8 const& v)
{
    u = u & v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator|=(int8& u, int8 const& v)
{
    u = u | v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator^=(int8& u, int8 const& v)
{
    u = u ^ v;
    return u;
}

VSNRAY_FORCE_INLINE int8& operator<<=(int8& a, int count)
{
    a = a << count;
    return a;
}

VSNRAY_FORCE_INLINE int8& operator>>=(int8& a, int count)
{
    a = a >> count;
    return a;
}

VSNRAY_FORCE_INLINE mask8 operator<(int8 const& u, int8 const& v)
{
    return _mm256_cmp_ps(float8(u), float8(v), _CMP_LT_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator>(int8 const& u, int8 const& v)
{
    return _mm256_cmp_ps(reinterpret_as_float(u), reinterpret_as_float(v), _CMP_GT_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator==(int8 const& u, int8 const& v)
{
    return _mm256_cmp_ps(reinterpret_as_float(u), reinterpret_as_float(v), _CMP_EQ_OQ);
}



//-------------------------------------------------------------------------------------------------
// mask8
//

/* float8 */

VSNRAY_FORCE_INLINE mask8 operator<(float8 const& u, float8 const& v)
{
    return _mm256_cmp_ps(u, v, _CMP_LT_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator>(float8 const& u, float8 const& v)
{
    return _mm256_cmp_ps(u, v, _CMP_GT_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator<=(float8 const& u, float8 const& v)
{
    return _mm256_cmp_ps(u, v, _CMP_LE_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator>=(float8 const& u, float8 const& v)
{
    return _mm256_cmp_ps(u, v, _CMP_GE_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator==(float8 const& u, float8 const& v)
{
    return _mm256_cmp_ps(u, v, _CMP_EQ_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator!=(float8 const& u, float8 const& v)
{
    return _mm256_cmp_ps(u, v, _CMP_NEQ_OQ);
}

VSNRAY_FORCE_INLINE mask8 operator&&(float8 const& u, float8 const& v)
{
    return _mm256_and_ps(u, v);
}

VSNRAY_FORCE_INLINE mask8 operator||(float8 const& u, float8 const& v)
{
    return _mm256_or_ps(u, v);
}


VSNRAY_FORCE_INLINE bool any(mask8 const& m)
{
    return _mm256_movemask_ps(m.f) != 0;
}

VSNRAY_FORCE_INLINE bool all(mask8 const& m)
{
    return _mm256_movemask_ps(m.f) == 0xFF;
}

VSNRAY_FORCE_INLINE mask8 operator!(mask8 const& a)
{
    return _mm256_xor_ps(a.f, mask8(true).f);
}

VSNRAY_FORCE_INLINE mask8 operator&(mask8 const& a, mask8 const& b)
{
    return _mm256_and_ps(a.f, b.f);
}

VSNRAY_FORCE_INLINE mask8 operator|(mask8 const& a, mask8 const& b)
{
    return _mm256_or_ps(a.f, b.f);
}

VSNRAY_FORCE_INLINE mask8 operator^(mask8 const& a, mask8 const& b)
{
    return _mm256_xor_ps(a.f, b.f);
}

VSNRAY_FORCE_INLINE mask8& operator&=(mask8& a, mask8 const& b)
{
    a = a & b;
    return a;
}

VSNRAY_FORCE_INLINE mask8& operator|=(mask8& a, mask8 const& b)
{
    a = a | b;
    return a;
}

VSNRAY_FORCE_INLINE mask8& operator^=(mask8& a, mask8 const& b)
{
    a = a ^ b;
    return a;
}

template <typename S, typename T>
VSNRAY_FORCE_INLINE void store(S dst[8], T const& v, mask8 const& mask)
{
    T old(dst);
    store( dst, select(mask, v, old) );
}

template <typename S, typename T>
VSNRAY_FORCE_INLINE void store(S dst[8], T const& v, mask8 const& mask, T const& old)
{
    store( dst, select(mask, v, old) );
}

VSNRAY_FORCE_INLINE float8 sqrt(float8 const& v)
{
    return _mm256_sqrt_ps(v);
}


//-------------------------------------------------------------------------------------------------
// cstdlib-like functions
//

VSNRAY_FORCE_INLINE float8 min(float8 const& u, float8 const& v)
{
    return _mm256_min_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 max(float8 const& u, float8 const& v)
{
    return _mm256_max_ps(u, v);
}

VSNRAY_FORCE_INLINE float8 saturate(float8 const& u)
{
    return _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(u, _mm256_set1_ps(1.0f)));
}

VSNRAY_FORCE_INLINE float8 abs(float8 const& u)
{
    return _mm256_and_ps(u, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}

VSNRAY_FORCE_INLINE float8 ceil(float8 const& v)
{
    return _mm256_ceil_ps(v);
}

VSNRAY_FORCE_INLINE float8 floor(float8 const& v)
{
    return _mm256_floor_ps(v);
}


} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

#endif // VSNRAY_MATH_SIMD_AVX_H
