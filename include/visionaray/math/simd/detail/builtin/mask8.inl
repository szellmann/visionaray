// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask8 members
//

MATH_FUNC
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
    : value{x1, x2, x3, x4, x5, x6, x7, x8}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8::basic_mask(bool const v[8])
    : value{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8::basic_mask(bool b)
    : value{b, b, b, b, b, b, b, b}
{
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 convert_to_int(mask8 const& a)
{
    return int8(
            a.value[0],
            a.value[1],
            a.value[2],
            a.value[3],
            a.value[4],
            a.value[5],
            a.value[6],
            a.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

MATH_FUNC
VSNRAY_FORCE_INLINE bool any(mask8 const& m)
{
    return m.value[0] || m.value[1] || m.value[2] || m.value[3]
        || m.value[4] || m.value[5] || m.value[6] || m.value[7];
}

MATH_FUNC
VSNRAY_FORCE_INLINE bool all(mask8 const& m)
{
    return m.value[0] && m.value[1] && m.value[2] && m.value[3]
        && m.value[4] && m.value[5] && m.value[6] && m.value[7];
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 select(mask8 const& m, mask8 const& a, mask8 const& b)
{
    return mask8(
            m.value[0] ? a.value[0] : b.value[0],
            m.value[1] ? a.value[1] : b.value[1],
            m.value[2] ? a.value[2] : b.value[2],
            m.value[3] ? a.value[3] : b.value[3],
            m.value[4] ? a.value[4] : b.value[4],
            m.value[5] ? a.value[5] : b.value[5],
            m.value[6] ? a.value[6] : b.value[6],
            m.value[7] ? a.value[7] : b.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

template <typename S>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[8], mask8 const& m)
{
    dst[0] = S(m.value[0]);
    dst[1] = S(m.value[1]);
    dst[2] = S(m.value[2]);
    dst[3] = S(m.value[3]);
    dst[4] = S(m.value[4]);
    dst[5] = S(m.value[5]);
    dst[6] = S(m.value[6]);
    dst[7] = S(m.value[7]);
}

template <typename S, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[8], T const& v, mask8 const& mask)
{
    dst[0] = mask.value[0] ? S(v.value[0]) : dst[0];
    dst[1] = mask.value[1] ? S(v.value[1]) : dst[1];
    dst[2] = mask.value[2] ? S(v.value[2]) : dst[2];
    dst[3] = mask.value[3] ? S(v.value[3]) : dst[3];
    dst[4] = mask.value[4] ? S(v.value[4]) : dst[4];
    dst[5] = mask.value[5] ? S(v.value[5]) : dst[5];
    dst[6] = mask.value[6] ? S(v.value[6]) : dst[6];
    dst[7] = mask.value[7] ? S(v.value[7]) : dst[7];
}

template <typename S, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[8], T const& v, mask8 const& mask, T const& old)
{
    dst[0] = mask.value[0] ? S(v.value[0]) : S(old.value[0]);
    dst[1] = mask.value[1] ? S(v.value[1]) : S(old.value[1]);
    dst[2] = mask.value[2] ? S(v.value[2]) : S(old.value[2]);
    dst[3] = mask.value[3] ? S(v.value[3]) : S(old.value[3]);
    dst[4] = mask.value[4] ? S(v.value[4]) : S(old.value[4]);
    dst[5] = mask.value[5] ? S(v.value[5]) : S(old.value[5]);
    dst[6] = mask.value[6] ? S(v.value[6]) : S(old.value[6]);
    dst[7] = mask.value[7] ? S(v.value[7]) : S(old.value[7]);
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator!(mask8 const& a)
{
    return mask8(
            !a.value[0],
            !a.value[1],
            !a.value[2],
            !a.value[3],
            !a.value[4],
            !a.value[5],
            !a.value[6],
            !a.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator&(mask8 const& a, mask8 const& b)
{
    return mask8(
            a.value[0] & b.value[0],
            a.value[1] & b.value[1],
            a.value[2] & b.value[2],
            a.value[3] & b.value[3],
            a.value[4] & b.value[4],
            a.value[5] & b.value[5],
            a.value[6] & b.value[6],
            a.value[7] & b.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator|(mask8 const& a, mask8 const& b)
{
    return mask8(
            a.value[0] | b.value[0],
            a.value[1] | b.value[1],
            a.value[2] | b.value[2],
            a.value[3] | b.value[3],
            a.value[4] | b.value[4],
            a.value[5] | b.value[5],
            a.value[6] | b.value[6],
            a.value[7] | b.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator^(mask8 const& a, mask8 const& b)
{
    return mask8(
            a.value[0] ^ b.value[0],
            a.value[1] ^ b.value[1],
            a.value[2] ^ b.value[2],
            a.value[3] ^ b.value[3],
            a.value[4] ^ b.value[4],
            a.value[5] ^ b.value[5],
            a.value[6] ^ b.value[6],
            a.value[7] ^ b.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator&&(mask8 const& a, mask8 const& b)
{
    return mask8(
            a.value[0] && b.value[0],
            a.value[1] && b.value[1],
            a.value[2] && b.value[2],
            a.value[3] && b.value[3],
            a.value[4] && b.value[4],
            a.value[5] && b.value[5],
            a.value[6] && b.value[6],
            a.value[7] && b.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator||(mask8 const& a, mask8 const& b)
{
    return mask8(
            a.value[0] || b.value[0],
            a.value[1] || b.value[1],
            a.value[2] || b.value[2],
            a.value[3] || b.value[3],
            a.value[4] || b.value[4],
            a.value[5] || b.value[5],
            a.value[6] || b.value[6],
            a.value[7] || b.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator==(mask8 const& u, mask8 const& v)
{
    return mask8(
            u.value[0] == v.value[0],
            u.value[1] == v.value[1],
            u.value[2] == v.value[2],
            u.value[3] == v.value[3],
            u.value[4] == v.value[4],
            u.value[5] == v.value[5],
            u.value[6] == v.value[6],
            u.value[7] == v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator!=(mask8 const& u, mask8 const& v)
{
    return mask8(
            u.value[0] != v.value[0],
            u.value[1] != v.value[1],
            u.value[2] != v.value[2],
            u.value[3] != v.value[3],
            u.value[4] != v.value[4],
            u.value[5] != v.value[5],
            u.value[6] != v.value[6],
            u.value[7] != v.value[7]
            );
}

} // simd
} // MATH_NAMESPACE
