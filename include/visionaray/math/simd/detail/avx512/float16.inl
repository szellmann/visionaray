// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cmath>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float16
//

VSNRAY_FORCE_INLINE float16::basic_float(
        float  x1, float  x2, float  x3, float  x4,
        float  x5, float  x6, float  x7, float  x8,
        float  x9, float x10, float x11, float x12,
        float x13, float x14, float x15, float x16
        )
    : value(_mm512_set_ps(x16, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1))
{
}

VSNRAY_FORCE_INLINE float16::basic_float(float const v[16])
    : value(_mm512_load_ps(v))
{
}

VSNRAY_FORCE_INLINE float16::basic_float(float s)
    : value(_mm512_set1_ps(s))
{
}

VSNRAY_FORCE_INLINE float16::basic_float(__m512i const& i)
    : value(_mm512_cvtepi32_ps(i))
{
}

VSNRAY_FORCE_INLINE float16::basic_float(__m512 const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE float16::operator __m512() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE int16 reinterpret_as_int(float16 const& a)
{
    return _mm512_castps_si512(a);
}

//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int16 convert_to_int(float16 const& a)
{
    return _mm512_cvttps_epi32(a);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE float16 select(mask16 const& m, float16 const& a, float16 const& b)
{
    return _mm512_mask_blend_ps(m, b, a);
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

VSNRAY_FORCE_INLINE void store(float dst[16], float16 const& v)
{
    _mm512_store_ps(dst, v);
}

template <unsigned I>
VSNRAY_FORCE_INLINE float& get(float16& v)
{
    static_assert(I < 16, "Index out of range for SIMD vector access");

    return reinterpret_cast<float*>(&v)[I];
}

template <unsigned I>
VSNRAY_FORCE_INLINE float const& get(float16 const& v)
{
    static_assert(I < 16, "Index out of range for SIMD vector access");

    return reinterpret_cast<float const*>(&v)[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetic
//

VSNRAY_FORCE_INLINE float16 operator+(float16 const& v)
{
    return _mm512_add_ps(_mm512_setzero_ps(), v);
}

VSNRAY_FORCE_INLINE float16 operator-(float16 const& v)
{
    return _mm512_sub_ps(_mm512_setzero_ps(), v);
}

VSNRAY_FORCE_INLINE float16 operator+(float16 const& u, float16 const& v)
{
    return _mm512_add_ps(u, v);
}

VSNRAY_FORCE_INLINE float16 operator-(float16 const& u, float16 const& v)
{
    return _mm512_sub_ps(u, v);
}

VSNRAY_FORCE_INLINE float16 operator*(float16 const& u, float16 const& v)
{
    return _mm512_mul_ps(u, v);
}

VSNRAY_FORCE_INLINE float16 operator/(float16 const& u, float16 const& v)
{
    return _mm512_div_ps(u, v);
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE float16 operator&(float16 const& u, float16 const& v)
{
    // TODO: available w/ AVX512DQ
//  return _mm512_and_ps(u, v);
    auto ui = reinterpret_as_int(u);
    auto vi = reinterpret_as_int(v);
    auto uandv = _mm512_and_epi32(ui, vi);
    return _mm512_castsi512_ps(uandv);
}

VSNRAY_FORCE_INLINE float16 operator|(float16 const& u, float16 const& v)
{
    // TODO: available w/ AVX512DQ
//  return _mm512_or_ps(u, v);
    auto ui = reinterpret_as_int(u);
    auto vi = reinterpret_as_int(v);
    auto uorv = _mm512_or_epi32(ui, vi);
    return _mm512_castsi512_ps(uorv);
}

VSNRAY_FORCE_INLINE float16 operator^(float16 const& u, float16 const& v)
{
    // TODO: available w/ AVX512DQ
//  return _mm512_xor_ps(u, v);
    auto ui = reinterpret_as_int(u);
    auto vi = reinterpret_as_int(v);
    auto uxorv = _mm512_xor_epi32(ui, vi);
    return _mm512_castsi512_ps(uxorv);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE float16 operator&&(float16 const& u, float16 const& v)
{
    return _mm512_and_ps(u, v);
}

VSNRAY_FORCE_INLINE float16 operator||(float16 const& u, float16 const& v)
{
    return _mm512_or_ps(u, v);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask16 operator<(float16 const& u, float16 const& v)
{
    return _mm512_cmp_ps_mask(u, v, _CMP_LT_OQ);
}

VSNRAY_FORCE_INLINE mask16 operator>(float16 const& u, float16 const& v)
{
    return _mm512_cmp_ps_mask(u, v, _CMP_GT_OQ);
}

VSNRAY_FORCE_INLINE mask16 operator<=(float16 const& u, float16 const& v)
{
    return _mm512_cmp_ps_mask(u, v, _CMP_LE_OQ);
}

VSNRAY_FORCE_INLINE mask16 operator>=(float16 const& u, float16 const& v)
{
    return _mm512_cmp_ps_mask(u, v, _CMP_GE_OQ);
}

VSNRAY_FORCE_INLINE mask16 operator==(float16 const& u, float16 const& v)
{
    return _mm512_cmp_ps_mask(u, v, _CMP_EQ_OQ);
}

VSNRAY_FORCE_INLINE mask16 operator!=(float16 const& u, float16 const& v)
{
    return _mm512_cmp_ps_mask(u, v, _CMP_NEQ_OQ);
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE float16 min(float16 const& u, float16 const& v)
{
    return _mm512_min_ps(u, v);
}

VSNRAY_FORCE_INLINE float16 max(float16 const& u, float16 const& v)
{
    return _mm512_max_ps(u, v);
}

VSNRAY_FORCE_INLINE float16 saturate(float16 const& u)
{
    return _mm512_max_ps(_mm512_setzero_ps(), _mm512_min_ps(u, _mm512_set1_ps(1.0f)));
}

VSNRAY_FORCE_INLINE float16 abs(float16 const& u)
{
//  return _mm512_abs_ps(u);
//  return _mm512_and_ps(u, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    auto ui = reinterpret_as_int(u);
    auto sign_mask = _mm512_set1_epi32(0x7FFFFFFF);
    auto a = _mm512_and_epi32(ui, sign_mask);
    return _mm512_castsi512_ps(a);
}

VSNRAY_FORCE_INLINE float16 ceil(float16 const& v)
{
    return _mm512_ceil_ps(v);
}

VSNRAY_FORCE_INLINE float16 floor(float16 const& v)
{
    return _mm512_floor_ps(v);
}

VSNRAY_FORCE_INLINE float16 sqrt(float16 const& v)
{
    return _mm512_sqrt_ps(v);
}

VSNRAY_FORCE_INLINE mask16 isinf(float16 const& v)
{
    VSNRAY_ALIGN(64) float values[16] = {};
    store(values, v);

    return mask16(
            std::isinf(values[ 0]),
            std::isinf(values[ 1]),
            std::isinf(values[ 2]),
            std::isinf(values[ 3]),
            std::isinf(values[ 4]),
            std::isinf(values[ 5]),
            std::isinf(values[ 6]),
            std::isinf(values[ 7]),
            std::isinf(values[ 8]),
            std::isinf(values[ 9]),
            std::isinf(values[10]),
            std::isinf(values[11]),
            std::isinf(values[12]),
            std::isinf(values[13]),
            std::isinf(values[14]),
            std::isinf(values[15])
            );
}

VSNRAY_FORCE_INLINE mask16 isnan(float16 const& v)
{
    return _mm512_cmp_ps_mask(v, v, _CMP_NEQ_UQ); // v != v unordered
}

VSNRAY_FORCE_INLINE mask16 isfinite(float16 const& v)
{
    return !(isinf(v) | isnan(v));
}

} // simd
} // MATH_NAMESPACE
