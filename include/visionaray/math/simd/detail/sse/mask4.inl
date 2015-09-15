// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask4 members
//

VSNRAY_FORCE_INLINE mask4::basic_mask(__m128 m)
    : f(m)
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(__m128i m)
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

VSNRAY_FORCE_INLINE mask4::basic_mask(bool b)
    : i( basic_int<__m128i>(b ? 0xFFFFFFFF : 0x00000000) )
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(basic_float<__m128> const& m)
    : f(m)
{
}

VSNRAY_FORCE_INLINE mask4::operator basic_float<__m128>() const
{
    return f;
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
// Logical operations
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

VSNRAY_FORCE_INLINE mask4& operator&=(mask4& a, mask4 const& b)
{
    a = a & b;
    return a;
}

VSNRAY_FORCE_INLINE mask4& operator|=(mask4& a, mask4 const& b)
{
    a = a | b;
    return a;
}

VSNRAY_FORCE_INLINE mask4& operator^=(mask4& a, mask4 const& b)
{
    a = a ^ b;
    return a;
}

} // simd
} // MATH_NAMESPACE
