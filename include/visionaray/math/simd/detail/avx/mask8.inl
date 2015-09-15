// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask8 members
//

VSNRAY_FORCE_INLINE mask8::basic_mask(__m256 m)
    : f(m)
{
}

VSNRAY_FORCE_INLINE mask8::basic_mask(__m256i m)
    : i(m)
{
}

VSNRAY_FORCE_INLINE mask8::basic_mask(bool b)
    : i( basic_int<__m256i>(b ? 0xFFFFFFFF : 0x00000000) )
{
}

VSNRAY_FORCE_INLINE mask8::basic_mask(
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

VSNRAY_FORCE_INLINE mask8::basic_mask(basic_float<__m256> const& m)
    : f(m)
{
}

VSNRAY_FORCE_INLINE mask8::operator basic_float<__m256>() const
{
    return f;
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

VSNRAY_FORCE_INLINE bool any(mask8 const& m)
{
    return _mm256_movemask_ps(m.f) != 0;
}

VSNRAY_FORCE_INLINE bool all(mask8 const& m)
{
    return _mm256_movemask_ps(m.f) == 0xFF;
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

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


//-------------------------------------------------------------------------------------------------
// Logical operations
//

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

} // simd
} // MATH_NAMESPACE
