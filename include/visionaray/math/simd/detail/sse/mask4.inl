// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask4 members
//

VSNRAY_FORCE_INLINE mask4::basic_mask(__m128 const& m)
    : f(m)
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(__m128i const& m)
    : i(m)
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(bool x, bool y, bool z, bool w)
    : i( basic_int<__m128i>(
            x ? 0xFFFFFFFF : 0x00000000,
            y ? 0xFFFFFFFF : 0x00000000,
            z ? 0xFFFFFFFF : 0x00000000,
            w ? 0xFFFFFFFF : 0x00000000
            ) )
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(bool const v[4])
    : i( basic_int<__m128i>(
            v[0] ? 0xFFFFFFFF : 0x00000000,
            v[1] ? 0xFFFFFFFF : 0x00000000,
            v[2] ? 0xFFFFFFFF : 0x00000000,
            v[3] ? 0xFFFFFFFF : 0x00000000
            ) )
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(bool b)
    : i( basic_int<__m128i>(b ? 0xFFFFFFFF : 0x00000000) )
{
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int4 convert_to_int(mask4 const& a)
{
    return a.i;
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

VSNRAY_FORCE_INLINE bool any(mask4 const& m)
{
    return _mm_movemask_ps(m.f) != 0;
}

VSNRAY_FORCE_INLINE bool all(mask4 const& m)
{
    return _mm_movemask_ps(m.f) == 0xF;
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE mask4 select(mask4 const& m, mask4 const& a, mask4 const& b)
{
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE4_1)
    return _mm_blendv_ps(b.f, a.f, m.f);
#else
    return _mm_or_ps(_mm_and_ps(m.f, a.f), _mm_andnot_ps(m.f, b.f));
#endif
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

template <typename S, typename T>
VSNRAY_FORCE_INLINE void store(S dst[4], T const& v, mask4 const& mask)
{
    T old(dst);
    store( dst, select(mask, v, old) );
}

template <typename S, typename T>
VSNRAY_FORCE_INLINE void store(S dst[4], T const& v, mask4 const& mask, T const& old)
{
    store( dst, select(mask, v, old) );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE mask4 operator!(mask4 const& a)
{
    return _mm_xor_si128(a.i, _mm_cmpeq_epi32(a.i, a.i));
}

VSNRAY_FORCE_INLINE mask4 operator&(mask4 const& a, mask4 const& b)
{
    return _mm_and_si128(a.i, b.i);
}

VSNRAY_FORCE_INLINE mask4 operator|(mask4 const& a, mask4 const& b)
{
    return _mm_or_si128(a.i, b.i);
}

VSNRAY_FORCE_INLINE mask4 operator^(mask4 const& a, mask4 const& b)
{
    return _mm_xor_si128(a.i, b.i);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE mask4 operator&&(mask4 const& a, mask4 const& b)
{
    // Ok because masks only store booleans
    return a & b;
}

VSNRAY_FORCE_INLINE mask4 operator||(mask4 const& a, mask4 const& b)
{
    // Ok because masks only store booleans
    return a | b;
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator==(mask4 const& u, mask4 const& v)
{
    return _mm_cmpeq_epi32(u.i, v.i);
}

VSNRAY_FORCE_INLINE mask4 operator!=(mask4 const& u, mask4 const& v)
{
    return !(u == v);
}

} // simd
} // MATH_NAMESPACE
