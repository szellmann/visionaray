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
    : value(_mm_set_epi32(w, z, y, x))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(int const v[4])
    : value(_mm_load_si128(reinterpret_cast<__m128i const*>(v)))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(int s)
    : value(_mm_set1_epi32(s))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(unsigned s)
    : value(_mm_set1_epi32(s))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(basic_float<__m128> const& f)
    : value(_mm_cvttps_epi32(f))
{
}

VSNRAY_FORCE_INLINE int4::basic_int(__m128i const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE int4::operator __m128i() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE float4 reinterpret_as_float(int4 const& a)
{
    return _mm_castsi128_ps(a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE float4 convert_to_float(int4 const& a)
{
    return _mm_cvtepi32_ps(a);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE int4 select(mask4 const& m, int4 const& a, int4 const& b)
{
    return reinterpret_as_int( select(m, reinterpret_as_float(a), reinterpret_as_float(b)) );
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

VSNRAY_FORCE_INLINE void store(int dst[4], int4 const& v)
{
    _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

VSNRAY_FORCE_INLINE void store(unsigned dst[4], int4 const& v)
{
    _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

VSNRAY_FORCE_INLINE void store_non_temporal(int dst[4], int4 const& v)
{
    _mm_stream_si128(reinterpret_cast<__m128i*>(dst), v);
}

VSNRAY_FORCE_INLINE void store_non_temporal(unsigned dst[4], int4 const& v)
{
    _mm_stream_si128(reinterpret_cast<__m128i*>(dst), v);
}

template <int A0, int A1, int A2, int A3>
VSNRAY_FORCE_INLINE int4 shuffle(int4 const& a)
{
    return _mm_shuffle_epi32(a, _MM_SHUFFLE(A3, A2, A1, A0));
}

template <unsigned I>
VSNRAY_FORCE_INLINE int& get(int4& v)
{
    static_assert(I >= 0 && I < 4, "Index out of range for SIMD vector access");

    return reinterpret_cast<int*>(&v)[I];
}

template <unsigned I>
VSNRAY_FORCE_INLINE int const& get(int4 const& v)
{
    static_assert(I >= 0 && I < 4, "Index out of range for SIMD vector access");

    return reinterpret_cast<int const*>(&v)[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmethic
//

VSNRAY_FORCE_INLINE int4 operator+(int4 const& v)
{
    return _mm_add_epi32(_mm_setzero_si128(), v);
}

VSNRAY_FORCE_INLINE int4 operator-(int4 const& v)
{
    return _mm_sub_epi32(_mm_setzero_si128(), v);
}

VSNRAY_FORCE_INLINE int4 operator+(int4 const& u, int4 const& v)
{
    return _mm_add_epi32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator-(int4 const& u, int4 const& v)
{
    return _mm_sub_epi32(u, v);
}

VSNRAY_FORCE_INLINE int4 operator*(int4 const& u, int4 const& v)
{
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE4_1)
    return _mm_mullo_epi32(u, v);
#else
    __m128i t0 = shuffle<1,0,3,0>(u);             // a1  ... a3  ...
    __m128i t1 = shuffle<1,0,3,0>(v);             // b1  ... b3  ...
    __m128i t2 = _mm_mul_epu32(u, v);             // ab0 ... ab2 ...
    __m128i t3 = _mm_mul_epu32(t0, t1);           // ab1 ... ab3 ...
    __m128i t4 = _mm_unpacklo_epi32(t2, t3);      // ab0 ab1 ... ...
    __m128i t5 = _mm_unpackhi_epi32(t2, t3);      // ab2 ab3 ... ...
    __m128i t6 = _mm_unpacklo_epi64(t4, t5);      // ab0 ab1 ab2 ab3

    return t6;
#endif
}

VSNRAY_FORCE_INLINE int4 operator/(int4 const& u, int4 const& v)
{
    return convert_to_int(convert_to_float(u) / convert_to_float(v));
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
    return _mm_and_si128(u, v);
}

VSNRAY_FORCE_INLINE int4 operator|(int4 const& u, int4 const& v)
{
    return _mm_or_si128(u, v);
}

VSNRAY_FORCE_INLINE int4 operator^(int4 const& u, int4 const& v)
{
    return _mm_xor_si128(u, v);
}

VSNRAY_FORCE_INLINE int4 operator<<(int4 const& a, int count)
{
    return _mm_slli_epi32(a, count);
}

VSNRAY_FORCE_INLINE int4 operator>>(int4 const& a, int count)
{
    return _mm_srli_epi32(a, count);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE int4 operator&&(int4 const& u, int4 const& v)
{
    return _mm_and_si128(u, v);
}

VSNRAY_FORCE_INLINE int4 operator||(int4 const& u, int4 const& v)
{
    return _mm_or_si128(u, v);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator<(int4 const& u, int4 const& v)
{
    return _mm_cmplt_epi32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>(int4 const& u, int4 const& v)
{
    return _mm_cmpgt_epi32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator==(int4 const& u, int4 const& v)
{
    return _mm_cmpeq_epi32(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator<=(int4 const& u, int4 const& v)
{
    return u < v || u == v;
}

VSNRAY_FORCE_INLINE mask4 operator>=(int4 const& u, int4 const& v)
{
    return u > v || u == v;
}

VSNRAY_FORCE_INLINE mask4 operator!=(int4 const& u, int4 const& v)
{
    return !(u == v);
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE int4 min(int4 const& u, int4 const& v)
{
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE4_1)
    return _mm_min_epi32(u, v);
#else
    return select(mask4(u < v), u, v);
#endif
}

VSNRAY_FORCE_INLINE int4 max(int4 const& u, int4 const& v)
{
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE4_1)
    return _mm_max_epi32(u, v);
#else
    return select(mask4(u > v), u, v);
#endif
}

} // simd
} // MATH_NAMESPACE
