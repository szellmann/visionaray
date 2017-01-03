// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// int16
//

VSNRAY_FORCE_INLINE int16::basic_int(
        int  x1, int  x2, int  x3, int  x4,
        int  x5, int  x6, int  x7, int  x8,
        int  x9, int x10, int x11, int x12,
        int x13, int x14, int x15, int x16
        )
    : value(_mm512_set_epi32(x16, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1))
{
}

VSNRAY_FORCE_INLINE int16::basic_int(int const v[16])
    : value(_mm512_load_si512(reinterpret_cast<__m512i const*>(v)))
{
}

VSNRAY_FORCE_INLINE int16::basic_int(int s)
    : value(_mm512_set1_epi32(s))
{
}

VSNRAY_FORCE_INLINE int16::basic_int(unsigned s)
    : value(_mm512_set1_epi32(s))
{
}

VSNRAY_FORCE_INLINE int16::basic_int(basic_float<__m512> const& f)
    : value(_mm512_cvttps_epi32(f))
{
}

VSNRAY_FORCE_INLINE int16::basic_int(__m512i const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE int16::operator __m512i() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE float16 reinterpret_as_float(int16 const& a)
{
    return _mm512_castsi512_ps(a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE float16 convert_to_float(int16 const& a)
{
    return _mm512_cvtepi32_ps(a);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE int16 select(mask16 const& m, int16 const& a, int16 const& b)
{
    return _mm512_mask_blend_epi32(m, a, b);
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

VSNRAY_FORCE_INLINE void store(int dst[16], int16 const& v)
{
    return _mm512_store_si512(reinterpret_cast<__m512i*>(dst), v);
}

VSNRAY_FORCE_INLINE void store(unsigned dst[16], int16 const& v)
{
    return _mm512_store_si512(reinterpret_cast<__m512i*>(dst), v);
}

template <size_t I>
VSNRAY_FORCE_INLINE int& get(int16& v)
{
    static_assert(I >= 0 && I < 16, "Index out of range for SIMD vector access");

    return reinterpret_cast<int*>(&v)[I];
}

template <size_t I>
VSNRAY_FORCE_INLINE int const& get(int16 const& v)
{
    static_assert(I >= 0 && I < 16, "Index out of range for SIMD vector access");

    return reinterpret_cast<int const*>(&v)[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetic
//

VSNRAY_FORCE_INLINE int16 operator+(int16 const& v)
{
    return _mm512_add_epi32(_mm512_setzero_si512(), v);
}

VSNRAY_FORCE_INLINE int16 operator-(int16 const& v)
{
    return _mm512_sub_epi32(_mm512_setzero_si512(), v);
}

VSNRAY_FORCE_INLINE int16 operator+(int16 const& u, int16 const& v)
{
    return _mm512_add_epi32(u, v);
}

VSNRAY_FORCE_INLINE int16 operator-(int16 const& u, int16 const& v)
{
    return _mm512_sub_epi32(u, v);
}

VSNRAY_FORCE_INLINE int16 operator*(int16 const& u, int16 const& v)
{
    return _mm512_mul_epi32(u, v);
}

VSNRAY_FORCE_INLINE int16 operator/(int16 const& u, int16 const& v)
{
    return reinterpret_as_int(reinterpret_as_float(u) / reinterpret_as_float(v));
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE int16 operator&(int16 const& u, int16 const& v)
{
    return _mm512_and_si512(u, v);
}

VSNRAY_FORCE_INLINE int16 operator|(int16 const& u, int16 const& v)
{
    return _mm512_or_si512(u, v);
}

VSNRAY_FORCE_INLINE int16 operator^(int16 const& u, int16 const& v)
{
    return _mm512_xor_si512(u, v);
}

VSNRAY_FORCE_INLINE int16 operator<<(int16 const& a, int count)
{
    return _mm512_slli_epi32(a, count);
}

VSNRAY_FORCE_INLINE int16 operator>>(int16 const& a, int count)
{
    return _mm512_srai_epi32(a, count);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE int16 operator&&(int16 const& u, int16 const& v)
{
    return _mm512_and_epi32(u, v);
}

VSNRAY_FORCE_INLINE int16 operator||(int16 const& u, int16 const& v)
{
    return _mm512_or_epi32(u, v);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask16 operator<(int16 const& u, int16 const& v)
{
    return _mm512_cmp_epi32_mask(u, v, _MM_CMPINT_LT);
}

VSNRAY_FORCE_INLINE mask16 operator>(int16 const& u, int16 const& v)
{
    return _mm512_cmp_epi32_mask(u, v, _MM_CMPINT_GT);
}

VSNRAY_FORCE_INLINE mask16 operator<=(int16 const& u, int16 const& v)
{
    return _mm512_cmp_epi32_mask(u, v, _MM_CMPINT_LE);
}

VSNRAY_FORCE_INLINE mask16 operator>=(int16 const& u, int16 const& v)
{
    return _mm512_cmp_epi32_mask(u, v, _MM_CMPINT_GE);
}

VSNRAY_FORCE_INLINE mask16 operator==(int16 const& u, int16 const& v)
{
    return _mm512_cmp_epi32_mask(u, v, _MM_CMPINT_EQ);
}

VSNRAY_FORCE_INLINE mask16 operator!=(int16 const& u, int16 const& v)
{
    //return _mm512_cmp_epi32_mask(u, v, _MM_CMPINT_NEQ);
    return !(u == v);
}

} // simd
} // MATH_NAMESPACE
