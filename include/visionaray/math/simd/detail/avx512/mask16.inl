// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask16 members
//

VSNRAY_FORCE_INLINE mask16::basic_mask(__mmask16 const& m)
    : value(m)
{
}

VSNRAY_FORCE_INLINE mask16::basic_mask(bool b)
    : value(b ? 0xFFFF : 0x0000)
{
}

VSNRAY_FORCE_INLINE mask16::basic_mask(
        bool  x1, bool  x2, bool  x3, bool  x4,
        bool  x5, bool  x6, bool  x7, bool  x8,
        bool  x9, bool x10, bool x11, bool x12,
        bool x13, bool x14, bool x15, bool x16
        )
    : value(
        (0x1    &  x1) | (0x2    &  x2) | (0x4    &  x3) | (0x8    &  x4) |
        (0x10   &  x5) | (0x20   &  x6) | (0x40   &  x7) | (0x80   &  x8) |
        (0x100  &  x9) | (0x200  & x10) | (0x400  & x11) | (0x800  & x12) |
        (0x1000 & x13) | (0x2000 & x14) | (0x4000 & x15) | (0x8000 & x16)
        )
{
}

VSNRAY_FORCE_INLINE mask16::basic_mask(bool const v[16])
    : value(
        (0x1    & v[ 0]) | (0x2    & v[ 1]) | (0x4    & v[ 2]) | (0x8    & v[ 3]) |
        (0x10   & v[ 4]) | (0x20   & v[ 5]) | (0x40   & v[ 6]) | (0x80   & v[ 7]) |
        (0x100  & v[ 8]) | (0x200  & v[ 9]) | (0x400  & v[10]) | (0x800  & v[11]) |
        (0x1000 & v[12]) | (0x2000 & v[13]) | (0x4000 & v[14]) | (0x8000 & v[15])
        )
{
}

VSNRAY_FORCE_INLINE mask16::operator __mmask16() const
{
    return value;
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

VSNRAY_FORCE_INLINE bool any(mask16 const& m)
{
    return m.value != 0;
}

VSNRAY_FORCE_INLINE bool all(mask16 const& m)
{
    return m.value == 0xFFFF;
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE mask16 select(mask16 const& m, mask16 const& a, mask16 const& b)
{
    return mask16(
            m.value & 0x1    ? bool(a & 0x1)    : bool(b & 0x1),
            m.value & 0x2    ? bool(a & 0x2)    : bool(b & 0x2),
            m.value & 0x4    ? bool(a & 0x4)    : bool(b & 0x4),
            m.value & 0x8    ? bool(a & 0x8)    : bool(b & 0x8),
            m.value & 0x10   ? bool(a & 0x10)   : bool(b & 0x10),
            m.value & 0x20   ? bool(a & 0x20)   : bool(b & 0x20),
            m.value & 0x40   ? bool(a & 0x40)   : bool(b & 0x40),
            m.value & 0x80   ? bool(a & 0x80)   : bool(b & 0x80),
            m.value & 0x100  ? bool(a & 0x100)  : bool(b & 0x100),
            m.value & 0x200  ? bool(a & 0x200)  : bool(b & 0x200),
            m.value & 0x400  ? bool(a & 0x400)  : bool(b & 0x400),
            m.value & 0x800  ? bool(a & 0x800)  : bool(b & 0x800),
            m.value & 0x1000 ? bool(a & 0x1000) : bool(b & 0x1000),
            m.value & 0x2000 ? bool(a & 0x2000) : bool(b & 0x2000),
            m.value & 0x4000 ? bool(a & 0x4000) : bool(b & 0x4000),
            m.value & 0x8000 ? bool(a & 0x8000) : bool(b & 0x8000)
            );
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

template <
    typename S,
    typename = typename std::enable_if<std::is_integral<S>::value>::type
    >
VSNRAY_FORCE_INLINE void store(S dst[16], mask16 const& mask)
{
    dst[ 0] = mask.value & 0x1    ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 1] = mask.value & 0x2    ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 2] = mask.value & 0x4    ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 3] = mask.value & 0x8    ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 4] = mask.value & 0x10   ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 5] = mask.value & 0x20   ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 6] = mask.value & 0x40   ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 7] = mask.value & 0x80   ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 8] = mask.value & 0x100  ? S(0xFFFFFFFF) : S(0x00000000);
    dst[ 9] = mask.value & 0x200  ? S(0xFFFFFFFF) : S(0x00000000);
    dst[10] = mask.value & 0x400  ? S(0xFFFFFFFF) : S(0x00000000);
    dst[11] = mask.value & 0x800  ? S(0xFFFFFFFF) : S(0x00000000);
    dst[12] = mask.value & 0x1000 ? S(0xFFFFFFFF) : S(0x00000000);
    dst[13] = mask.value & 0x2000 ? S(0xFFFFFFFF) : S(0x00000000);
    dst[14] = mask.value & 0x4000 ? S(0xFFFFFFFF) : S(0x00000000);
    dst[15] = mask.value & 0x8000 ? S(0xFFFFFFFF) : S(0x00000000);
}

// TODO: the following don't necessarily store masks!
// TODO: FIXME for SSE and AVX!
template <typename S, typename T>
VSNRAY_FORCE_INLINE void store(S dst[16], T const& v, mask16 const& mask)
{
    T old(dst);
    store( dst, select(mask, v, old) );
}

template <typename S, typename T>
VSNRAY_FORCE_INLINE void store(S dst[16], T const& v, mask16 const& mask, T const& old)
{
    store( dst, select(mask, v, old) );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE mask16 operator!(mask16 const& a)
{
    return _mm512_knot(a);
}

VSNRAY_FORCE_INLINE mask16 operator&(mask16 const& a, mask16 const& b)
{
    return _mm512_kand(a, b);
}

VSNRAY_FORCE_INLINE mask16 operator|(mask16 const& a, mask16 const& b)
{
    return _mm512_kor(a, b);
}

VSNRAY_FORCE_INLINE mask16 operator^(mask16 const& a, mask16 const& b)
{
    return _mm512_kxor(a, b);
}

VSNRAY_FORCE_INLINE mask16 operator&(mask16 const& a, bool b)
{
    return _mm512_kand(a, mask16(b));
}

VSNRAY_FORCE_INLINE mask16 operator|(mask16 const& a, bool b)
{
    return _mm512_kor(a, mask16(b));
}

VSNRAY_FORCE_INLINE mask16 operator^(mask16 const& a, bool b)
{
    return _mm512_kxor(a, mask16(b));
}

VSNRAY_FORCE_INLINE mask16 operator&(bool a, mask16 const& b)
{
    return _mm512_kand(mask16(a), b);
}

VSNRAY_FORCE_INLINE mask16 operator|(bool a, mask16 const& b)
{
    return _mm512_kor(mask16(a), b);
}

VSNRAY_FORCE_INLINE mask16 operator^(bool a, mask16 const& b)
{
    return _mm512_kxor(mask16(a), b);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE mask16 operator&&(mask16 const& a, mask16 const& b)
{
    // Ok because masks store bitfields
    return a & b;
}

VSNRAY_FORCE_INLINE mask16 operator||(mask16 const& a, mask16 const& b)
{
    // Ok because masks store bitfields
    return a | b;
}

VSNRAY_FORCE_INLINE mask16 operator&&(mask16 const& a, bool b)
{
    // Ok because masks store bitfields
    return a & b;
}

VSNRAY_FORCE_INLINE mask16 operator||(mask16 const& a, bool b)
{
    // Ok because masks store bitfields
    return a | b;
}

VSNRAY_FORCE_INLINE mask16 operator&&(bool a, mask16 const& b)
{
    // Ok because masks store bitfields
    return a & b;
}

VSNRAY_FORCE_INLINE mask16 operator||(bool a, mask16 const& b)
{
    // Ok because masks store bitfields
    return a | b;
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask16 operator==(mask16 const& u, mask16 const& v)
{
    return *reinterpret_cast<short const*>(&u.value) == *reinterpret_cast<short const*>(&v.value);
}

VSNRAY_FORCE_INLINE mask16 operator!=(mask16 const& u, mask16 const& v)
{
    return *reinterpret_cast<short const*>(&u.value) != *reinterpret_cast<short const*>(&v.value);
}

} // simd
} // MATH_NAMESPACE
