// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float4 members
//

VSNRAY_FORCE_INLINE float4::basic_float(float x, float y, float z, float w)
    : value(_mm_set_ps(w, z, y, x))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(float const v[4])
    : value(_mm_load_ps(v))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(float s)
    : value(_mm_set1_ps(s))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(__m128i const& i)
    : value(_mm_cvtepi32_ps(i))
{
}

VSNRAY_FORCE_INLINE float4::basic_float(__m128 const& v)
    : value(v)
{
}

VSNRAY_FORCE_INLINE float4::operator __m128() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

inline int4 reinterpret_as_int(float4 const& a)
{
    return _mm_castps_si128(a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

inline int4 convert_to_int(float4 const& a)
{
    return _mm_cvttps_epi32(a);
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE float4 select(mask4 const& m, float4 const& a, float4 const& b)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE4_1
    return _mm_blendv_ps(b, a, m.f);
#else
    return _mm_or_ps(_mm_and_ps(m.f, a), _mm_andnot_ps(m.f, b));
#endif
}

VSNRAY_FORCE_INLINE float4 select(mask4 const& m, float4 const& a, float b)
{
    return select(m, a, float4(b));
}

VSNRAY_FORCE_INLINE float4 select(mask4 const& m, float a, float4 const& b)
{
    return select(m, float4(a), b);
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

VSNRAY_FORCE_INLINE float4 load(float const dst[4])
{
    return _mm_load_ps(dst);
}

VSNRAY_FORCE_INLINE float4 load_unaligned(float const dst[4])
{
    return _mm_loadu_ps(dst);
}

VSNRAY_FORCE_INLINE void store(float dst[4], float4 const& v)
{
    _mm_store_ps(dst, v);
}

VSNRAY_FORCE_INLINE void store_unaligned(float dst[4], float4 const& v)
{
    _mm_storeu_ps(dst, v);
}

template <int U0, int U1, int V2, int V3>
VSNRAY_FORCE_INLINE float4 shuffle(float4 const& u, float4 const& v)
{
    return _mm_shuffle_ps(u, v, _MM_SHUFFLE(V3, V2, U1, U0));
}

template <int V0, int V1, int V2, int V3>
VSNRAY_FORCE_INLINE float4 shuffle(float4 const& v)
{
    return _mm_shuffle_ps(v, v, _MM_SHUFFLE(V3, V2, V1, V0));
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetics
//

VSNRAY_FORCE_INLINE float4 operator-(float4 const& v)
{
    return _mm_sub_ps(_mm_setzero_ps(), v);
}

VSNRAY_FORCE_INLINE float4 operator+(float4 const& u, float4 const& v)
{
    return _mm_add_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 operator-(float4 const& u, float4 const& v)
{
    return _mm_sub_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 operator*(float4 const& u, float4 const& v)
{
    return _mm_mul_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 operator/(float4 const& u, float4 const& v)
{
    return _mm_div_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 operator&(float4 const& u, float4 const& v)
{
    return _mm_and_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 operator|(float4 const& u, float4 const& v)
{
    return _mm_or_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 operator^(float4 const& u, float4 const& v)
{
    return _mm_xor_ps(u, v);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator<(float4 const& u, float4 const& v)
{
    return _mm_cmplt_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>(float4 const& u, float4 const& v)
{
    return _mm_cmpgt_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator<=(float4 const& u, float4 const& v)
{
    return _mm_cmple_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator>=(float4 const& u, float4 const& v)
{
    return _mm_cmpge_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator==(float4 const& u, float4 const& v)
{
    return _mm_cmpeq_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator!=(float4 const& u, float4 const& v)
{
    return _mm_cmpneq_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator&&(float4 const& u, float4 const& v)
{
    return _mm_and_ps(u, v);
}

VSNRAY_FORCE_INLINE mask4 operator||(float4 const& u, float4 const& v)
{
    return _mm_and_ps(u, v);
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

VSNRAY_FORCE_INLINE float4 dot(float4 const& u, float4 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE4_1
    return _mm_dp_ps(u, v, 0xFF);
#else
   __m128 t1 = _mm_mul_ps(u, v);
   __m128 t2 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2,3,0,1));
   __m128 t3 = _mm_add_ps(t1, t2);
   __m128 t4 = _mm_shuffle_ps(t3, t3, _MM_SHUFFLE(0,1,2,3));
   __m128 t5 = _mm_add_ps(t3, t4);

   return t5;
#endif
}

VSNRAY_FORCE_INLINE float4 min(float4 const& u, float4 const& v)
{
    return _mm_min_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 max(float4 const& u, float4 const& v)
{
    return _mm_max_ps(u, v);
}

VSNRAY_FORCE_INLINE float4 saturate(float4 const& u)
{
    return _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(u, _mm_set1_ps(1.0f)));
}

VSNRAY_FORCE_INLINE float4 abs(float4 const& u)
{
    return _mm_and_ps(u, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}

VSNRAY_FORCE_INLINE float4 round(float4 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE4_1
    return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
    // Mask out the signbits of v
    __m128 s = _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
    // Magic number: 2^23 with the signbits of v
    __m128 m = _mm_or_ps(s, _mm_castsi128_ps(_mm_set1_epi32(0x4B000000)));
    __m128 x = _mm_add_ps(v, m);
    __m128 y = _mm_sub_ps(x, m);

    return y;
#endif
}

VSNRAY_FORCE_INLINE float4 ceil(float4 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE4_1
    return _mm_ceil_ps(v);
#else
    // i = trunc(v)
    __m128 i = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
    // r = i < v ? i i + 1 : i
    __m128 t = _mm_cmplt_ps(i, v);
    __m128 d = _mm_cvtepi32_ps(_mm_castps_si128(t)); // mask to float: 0 -> 0.0f, 0xFFFFFFFF -> -1.0f
    __m128 r = _mm_sub_ps(i, d);

    return r;
#endif
}

VSNRAY_FORCE_INLINE float4 floor(float4 const& v)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE4_1
    return _mm_floor_ps(v);
#else
    // i = trunc(v)
    __m128 i = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
    // r = i > v ? i - 1 : i
    __m128 t = _mm_cmpgt_ps(i, v);
    __m128 d = _mm_cvtepi32_ps(_mm_castps_si128(t)); // mask to float: 0 -> 0.0f, 0xFFFFFFFF -> -1.0f
    __m128 r = _mm_add_ps(i, d);

    return r;
#endif
}

VSNRAY_FORCE_INLINE float4 sqrt(float4 const& v)
{
    return _mm_sqrt_ps(v);
}


//-------------------------------------------------------------------------------------------------
//
//

template <unsigned N>
VSNRAY_FORCE_INLINE float4 rcp_step(float4 const& v)
{
    float4 t = v;

    for (unsigned i = 0; i < N; ++i)
    {
        t = (t + t) - (v * t * t);
    }

    return t;
}

template <unsigned N>
VSNRAY_FORCE_INLINE float4 rcp(float4 const& v)
{
    float4 x0 = _mm_rcp_ps(v);
    return rcp_step<N>(x0);
}

VSNRAY_FORCE_INLINE float4 rcp(float4 const& v)
{
    float4 x0 = _mm_rcp_ps(v);
    return rcp_step<1>(x0);
}

template <unsigned N>
VSNRAY_FORCE_INLINE float4 rsqrt_step(float4 const& v, float4 const& x0)
{
    float4 threehalf(1.5f);
    float4 vhalf = v * float4(0.5f);
    float4 t = x0;

    for (unsigned i = 0; i < N; ++i)
    {
        t = t * (threehalf - vhalf * t * t);
    }

    return t;
}

template <unsigned N>
VSNRAY_FORCE_INLINE float4 rsqrt(float4 const& v)
{
    float4 x0 = _mm_rsqrt_ps(v);
    return rsqrt_step<N>(v, x0);
}

VSNRAY_FORCE_INLINE float4 rsqrt(float4 const& v)
{
    float4 x0 = _mm_rsqrt_ps(v);
    return rsqrt_step<1>(v, x0);
}

VSNRAY_FORCE_INLINE float4 approx_rsqrt(float4 const& v)
{
    return _mm_rsqrt_ps(v);
}

} // simd
} // MATH_NAMESPACE
