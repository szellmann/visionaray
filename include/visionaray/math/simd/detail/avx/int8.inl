// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// int8
//

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
    : value(_mm256_set_epi32(x8, x7, x6, x5, x4, x3, x2, x1))
{
}

VSNRAY_FORCE_INLINE int8::basic_int(int const v[8])
    : value(_mm256_load_si256(reinterpret_cast<__m256i const*>(v)))
{
}

VSNRAY_FORCE_INLINE int8::basic_int(int s)
    : value(_mm256_set1_epi32(s))
{
}

VSNRAY_FORCE_INLINE int8::basic_int(unsigned s)
    : value(_mm256_set1_epi32(s))
{
}

VSNRAY_FORCE_INLINE int8::basic_int(basic_float<__m256> const& f)
    : value(_mm256_cvtps_epi32(f))
{
}

VSNRAY_FORCE_INLINE int8::basic_int(__m256i const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE int8::operator __m256i() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

inline float8 reinterpret_as_float(int8 const& a)
{
    return _mm256_castsi256_ps(a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

// TODO


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE int8 select(mask8 const& m, int8 const& a, int8 const& b)
{
    return reinterpret_as_int( select(m, reinterpret_as_float(a), reinterpret_as_float(b)) );
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

VSNRAY_FORCE_INLINE void store(int dst[8], int8 const& v)
{
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
}

VSNRAY_FORCE_INLINE void store(unsigned dst[8], int8 const& v)
{
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetic
//

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


//-------------------------------------------------------------------------------------------------
// Bitwise operators
//

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


//-------------------------------------------------------------------------------------------------
// Comparisons
//

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

} // simd
} // MATH_NAMESPACE
