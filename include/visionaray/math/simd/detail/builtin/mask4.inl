// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask4 members
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4::basic_mask(bool x, bool y, bool z, bool w)
    : value{x, y, z, w}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4::basic_mask(bool const v[4])
    : value{v[0], v[1], v[2], v[3]}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4::basic_mask(bool b)
    : value{b, b, b, b}
{
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int4 convert_to_int(mask4 const& a)
{
    return int4(
            a.value[0],
            a.value[1],
            a.value[2],
            a.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

MATH_FUNC
VSNRAY_FORCE_INLINE bool any(mask4 const& m)
{
    return m.value[0] || m.value[1] || m.value[2] || m.value[3];
}

MATH_FUNC
VSNRAY_FORCE_INLINE bool all(mask4 const& m)
{
    return m.value[0] && m.value[1] && m.value[2] && m.value[3];
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 select(mask4 const& m, mask4 const& a, mask4 const& b)
{
    return mask4(
            m.value[0] ? a.value[0] : b.value[0],
            m.value[1] ? a.value[1] : b.value[1],
            m.value[2] ? a.value[2] : b.value[2],
            m.value[3] ? a.value[3] : b.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

template <typename S>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[4], mask4 const& m)
{
    dst[0] = S(m.value[0]);
    dst[1] = S(m.value[1]);
    dst[2] = S(m.value[2]);
    dst[3] = S(m.value[3]);
}

template <typename S, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[4], T const& v, mask4 const& mask)
{
    dst[0] = mask.value[0] ? S(v.value[0]) : dst[0];
    dst[1] = mask.value[1] ? S(v.value[1]) : dst[1];
    dst[2] = mask.value[2] ? S(v.value[2]) : dst[2];
    dst[3] = mask.value[3] ? S(v.value[3]) : dst[3];
}

template <typename S, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[4], T const& v, mask4 const& mask, T const& old)
{
    dst[0] = mask.value[0] ? S(v.value[0]) : S(old.value[0]);
    dst[1] = mask.value[1] ? S(v.value[1]) : S(old.value[1]);
    dst[2] = mask.value[2] ? S(v.value[2]) : S(old.value[2]);
    dst[3] = mask.value[3] ? S(v.value[3]) : S(old.value[3]);
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator!(mask4 const& a)
{
    return mask4(
            !a.value[0],
            !a.value[1],
            !a.value[2],
            !a.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator&(mask4 const& a, mask4 const& b)
{
    return mask4(
            a.value[0] & b.value[0],
            a.value[1] & b.value[1],
            a.value[2] & b.value[2],
            a.value[3] & b.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator|(mask4 const& a, mask4 const& b)
{
    return mask4(
            a.value[0] | b.value[0],
            a.value[1] | b.value[1],
            a.value[2] | b.value[2],
            a.value[3] | b.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator^(mask4 const& a, mask4 const& b)
{
    return mask4(
            a.value[0] ^ b.value[0],
            a.value[1] ^ b.value[1],
            a.value[2] ^ b.value[2],
            a.value[3] ^ b.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator&&(mask4 const& a, mask4 const& b)
{
    return mask4(
            a.value[0] && b.value[0],
            a.value[1] && b.value[1],
            a.value[2] && b.value[2],
            a.value[3] && b.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator||(mask4 const& a, mask4 const& b)
{
    return mask4(
            a.value[0] || b.value[0],
            a.value[1] || b.value[1],
            a.value[2] || b.value[2],
            a.value[3] || b.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator==(mask4 const& u, mask4 const& v)
{
    return mask4(
            u.value[0] == v.value[0],
            u.value[1] == v.value[1],
            u.value[2] == v.value[2],
            u.value[3] == v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator!=(mask4 const& u, mask4 const& v)
{
    return mask4(
            u.value[0] != v.value[0],
            u.value[1] != v.value[1],
            u.value[2] != v.value[2],
            u.value[3] != v.value[3]
            );
}

} // simd
} // MATH_NAMESPACE
