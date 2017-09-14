// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask16 members
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask16::basic_mask(
        bool  x1, bool  x2, bool  x3, bool  x4,
        bool  x5, bool  x6, bool  x7, bool  x8,
        bool  x9, bool x10, bool x11, bool x12,
        bool x13, bool x14, bool x15, bool x16
        )
    : value{
         x1,  x2,  x3,  x4,
         x5,  x6,  x7,  x8,
         x9, x10, x11, x12,
        x13, x14, x15, x16
        }
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16::basic_mask(bool const v[16])
    : value{
         v[0],  v[1],  v[2],  v[3],
         v[4],  v[5],  v[6],  v[7],
         v[8],  v[9], v[10], v[11],
        v[12], v[13], v[14], v[15]
        }
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16::basic_mask(bool b)
    : value{b, b, b, b, b, b, b, b, b, b, b, b, b, b, b}
{
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int16 convert_to_int(mask16 const& a)
{
    return int16(
            a.value[ 0],
            a.value[ 1],
            a.value[ 2],
            a.value[ 3],
            a.value[ 4],
            a.value[ 5],
            a.value[ 6],
            a.value[ 7],
            a.value[ 8],
            a.value[ 9],
            a.value[10],
            a.value[11],
            a.value[12],
            a.value[13],
            a.value[14],
            a.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

MATH_FUNC
VSNRAY_FORCE_INLINE bool any(mask16 const& m)
{
    return m.value[ 0] || m.value[ 1] || m.value[ 2] || m.value[ 3]
        || m.value[ 4] || m.value[ 5] || m.value[ 6] || m.value[ 7]
        || m.value[ 8] || m.value[ 9] || m.value[10] || m.value[11]
        || m.value[12] || m.value[13] || m.value[14] || m.value[15];
}

MATH_FUNC
VSNRAY_FORCE_INLINE bool all(mask16 const& m)
{
    return m.value[ 0] && m.value[ 1] && m.value[ 2] && m.value[ 3]
        && m.value[ 4] && m.value[ 5] && m.value[ 6] && m.value[ 7]
        && m.value[ 8] && m.value[ 9] && m.value[10] && m.value[11]
        && m.value[12] && m.value[13] && m.value[14] && m.value[15];
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 select(mask16 const& m, mask16 const& a, mask16 const& b)
{
    return mask16(
            m.value[ 0] ? a.value[ 0] : b.value[ 0],
            m.value[ 1] ? a.value[ 1] : b.value[ 1],
            m.value[ 2] ? a.value[ 2] : b.value[ 2],
            m.value[ 3] ? a.value[ 3] : b.value[ 3],
            m.value[ 4] ? a.value[ 4] : b.value[ 4],
            m.value[ 5] ? a.value[ 5] : b.value[ 5],
            m.value[ 6] ? a.value[ 6] : b.value[ 6],
            m.value[ 7] ? a.value[ 7] : b.value[ 7],
            m.value[ 8] ? a.value[ 8] : b.value[ 8],
            m.value[ 9] ? a.value[ 9] : b.value[ 9],
            m.value[10] ? a.value[10] : b.value[10],
            m.value[11] ? a.value[11] : b.value[11],
            m.value[12] ? a.value[12] : b.value[12],
            m.value[13] ? a.value[13] : b.value[13],
            m.value[14] ? a.value[14] : b.value[14],
            m.value[15] ? a.value[15] : b.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// Load / store
//

template <typename S>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[16], mask16 const& m)
{
    dst[ 0] = S(m.value[ 0]);
    dst[ 1] = S(m.value[ 1]);
    dst[ 2] = S(m.value[ 2]);
    dst[ 3] = S(m.value[ 3]);
    dst[ 4] = S(m.value[ 4]);
    dst[ 5] = S(m.value[ 5]);
    dst[ 6] = S(m.value[ 6]);
    dst[ 7] = S(m.value[ 7]);
    dst[ 8] = S(m.value[ 8]);
    dst[ 9] = S(m.value[ 9]);
    dst[10] = S(m.value[10]);
    dst[11] = S(m.value[11]);
    dst[12] = S(m.value[12]);
    dst[13] = S(m.value[13]);
    dst[14] = S(m.value[14]);
    dst[15] = S(m.value[15]);
}

template <typename S, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[16], T const& v, mask16 const& mask)
{
    dst[ 0] = mask.value[ 0] ? S(v.value[ 0]) : dst[ 0];
    dst[ 1] = mask.value[ 1] ? S(v.value[ 1]) : dst[ 1];
    dst[ 2] = mask.value[ 2] ? S(v.value[ 2]) : dst[ 2];
    dst[ 3] = mask.value[ 3] ? S(v.value[ 3]) : dst[ 3];
    dst[ 4] = mask.value[ 4] ? S(v.value[ 4]) : dst[ 4];
    dst[ 5] = mask.value[ 5] ? S(v.value[ 5]) : dst[ 5];
    dst[ 6] = mask.value[ 6] ? S(v.value[ 6]) : dst[ 6];
    dst[ 7] = mask.value[ 7] ? S(v.value[ 7]) : dst[ 7];
    dst[ 8] = mask.value[ 8] ? S(v.value[ 8]) : dst[ 8];
    dst[ 9] = mask.value[ 9] ? S(v.value[ 9]) : dst[ 9];
    dst[10] = mask.value[10] ? S(v.value[10]) : dst[10];
    dst[11] = mask.value[11] ? S(v.value[11]) : dst[11];
    dst[12] = mask.value[12] ? S(v.value[12]) : dst[12];
    dst[13] = mask.value[13] ? S(v.value[13]) : dst[13];
    dst[14] = mask.value[14] ? S(v.value[14]) : dst[14];
    dst[15] = mask.value[15] ? S(v.value[15]) : dst[15];
}

template <typename S, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE void store(S dst[16], T const& v, mask16 const& mask, T const& old)
{
    dst[ 0] = mask.value[ 0] ? S(v.value[ 0]) : S(old.value[ 0]);
    dst[ 1] = mask.value[ 1] ? S(v.value[ 1]) : S(old.value[ 1]);
    dst[ 2] = mask.value[ 2] ? S(v.value[ 2]) : S(old.value[ 2]);
    dst[ 3] = mask.value[ 3] ? S(v.value[ 3]) : S(old.value[ 3]);
    dst[ 4] = mask.value[ 4] ? S(v.value[ 4]) : S(old.value[ 4]);
    dst[ 5] = mask.value[ 5] ? S(v.value[ 5]) : S(old.value[ 5]);
    dst[ 6] = mask.value[ 6] ? S(v.value[ 6]) : S(old.value[ 6]);
    dst[ 7] = mask.value[ 7] ? S(v.value[ 7]) : S(old.value[ 7]);
    dst[ 8] = mask.value[ 8] ? S(v.value[ 8]) : S(old.value[ 8]);
    dst[ 9] = mask.value[ 9] ? S(v.value[ 9]) : S(old.value[ 9]);
    dst[10] = mask.value[10] ? S(v.value[10]) : S(old.value[10]);
    dst[11] = mask.value[11] ? S(v.value[11]) : S(old.value[11]);
    dst[12] = mask.value[12] ? S(v.value[12]) : S(old.value[12]);
    dst[13] = mask.value[13] ? S(v.value[13]) : S(old.value[13]);
    dst[14] = mask.value[14] ? S(v.value[14]) : S(old.value[14]);
    dst[15] = mask.value[15] ? S(v.value[15]) : S(old.value[15]);
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator!(mask16 const& a)
{
    return mask16(
            !a.value[ 0],
            !a.value[ 1],
            !a.value[ 2],
            !a.value[ 3],
            !a.value[ 4],
            !a.value[ 5],
            !a.value[ 6],
            !a.value[ 7],
            !a.value[ 8],
            !a.value[ 9],
            !a.value[10],
            !a.value[11],
            !a.value[12],
            !a.value[13],
            !a.value[14],
            !a.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator&(mask16 const& a, mask16 const& b)
{
    return mask16(
            a.value[ 0] & b.value[ 0],
            a.value[ 1] & b.value[ 1],
            a.value[ 2] & b.value[ 2],
            a.value[ 3] & b.value[ 3],
            a.value[ 4] & b.value[ 4],
            a.value[ 5] & b.value[ 5],
            a.value[ 6] & b.value[ 6],
            a.value[ 7] & b.value[ 7],
            a.value[ 8] & b.value[ 8],
            a.value[ 9] & b.value[ 9],
            a.value[10] & b.value[10],
            a.value[11] & b.value[11],
            a.value[12] & b.value[12],
            a.value[13] & b.value[13],
            a.value[14] & b.value[14],
            a.value[15] & b.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator|(mask16 const& a, mask16 const& b)
{
    return mask16(
            a.value[ 0] | b.value[ 0],
            a.value[ 1] | b.value[ 1],
            a.value[ 2] | b.value[ 2],
            a.value[ 3] | b.value[ 3],
            a.value[ 4] | b.value[ 4],
            a.value[ 5] | b.value[ 5],
            a.value[ 6] | b.value[ 6],
            a.value[ 7] | b.value[ 7],
            a.value[ 8] | b.value[ 8],
            a.value[ 9] | b.value[ 9],
            a.value[10] | b.value[10],
            a.value[11] | b.value[11],
            a.value[12] | b.value[12],
            a.value[13] | b.value[13],
            a.value[14] | b.value[14],
            a.value[15] | b.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator^(mask16 const& a, mask16 const& b)
{
    return mask16(
            a.value[ 0] ^ b.value[ 0],
            a.value[ 1] ^ b.value[ 1],
            a.value[ 2] ^ b.value[ 2],
            a.value[ 3] ^ b.value[ 3],
            a.value[ 4] ^ b.value[ 4],
            a.value[ 5] ^ b.value[ 5],
            a.value[ 6] ^ b.value[ 6],
            a.value[ 7] ^ b.value[ 7],
            a.value[ 8] ^ b.value[ 8],
            a.value[ 9] ^ b.value[ 9],
            a.value[10] ^ b.value[10],
            a.value[11] ^ b.value[11],
            a.value[12] ^ b.value[12],
            a.value[13] ^ b.value[13],
            a.value[14] ^ b.value[14],
            a.value[15] ^ b.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator&&(mask16 const& a, mask16 const& b)
{
    return mask16(
            a.value[ 0] && b.value[ 0],
            a.value[ 1] && b.value[ 1],
            a.value[ 2] && b.value[ 2],
            a.value[ 3] && b.value[ 3],
            a.value[ 4] && b.value[ 4],
            a.value[ 5] && b.value[ 5],
            a.value[ 6] && b.value[ 6],
            a.value[ 7] && b.value[ 7],
            a.value[ 8] && b.value[ 8],
            a.value[ 9] && b.value[ 9],
            a.value[10] && b.value[10],
            a.value[11] && b.value[11],
            a.value[12] && b.value[12],
            a.value[13] && b.value[13],
            a.value[14] && b.value[14],
            a.value[15] && b.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator||(mask16 const& a, mask16 const& b)
{
    return mask16(
            a.value[ 0] || b.value[ 0],
            a.value[ 1] || b.value[ 1],
            a.value[ 2] || b.value[ 2],
            a.value[ 3] || b.value[ 3],
            a.value[ 4] || b.value[ 4],
            a.value[ 5] || b.value[ 5],
            a.value[ 6] || b.value[ 6],
            a.value[ 7] || b.value[ 7],
            a.value[ 8] || b.value[ 8],
            a.value[ 9] || b.value[ 9],
            a.value[10] || b.value[10],
            a.value[11] || b.value[11],
            a.value[12] || b.value[12],
            a.value[13] || b.value[13],
            a.value[14] || b.value[14],
            a.value[15] || b.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator==(mask16 const& u, mask16 const& v)
{
    return mask16(
            u.value[ 0] == v.value[ 0],
            u.value[ 1] == v.value[ 1],
            u.value[ 2] == v.value[ 2],
            u.value[ 3] == v.value[ 3],
            u.value[ 4] == v.value[ 4],
            u.value[ 5] == v.value[ 5],
            u.value[ 6] == v.value[ 6],
            u.value[ 7] == v.value[ 7],
            u.value[ 8] == v.value[ 8],
            u.value[ 9] == v.value[ 9],
            u.value[10] == v.value[10],
            u.value[11] == v.value[11],
            u.value[12] == v.value[12],
            u.value[13] == v.value[13],
            u.value[14] == v.value[14],
            u.value[15] == v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator!=(mask16 const& u, mask16 const& v)
{
    return mask16(
            u.value[ 0] != v.value[ 0],
            u.value[ 1] != v.value[ 1],
            u.value[ 2] != v.value[ 2],
            u.value[ 3] != v.value[ 3],
            u.value[ 4] != v.value[ 4],
            u.value[ 5] != v.value[ 5],
            u.value[ 6] != v.value[ 6],
            u.value[ 7] != v.value[ 7],
            u.value[ 8] != v.value[ 8],
            u.value[ 9] != v.value[ 9],
            u.value[10] != v.value[10],
            u.value[11] != v.value[11],
            u.value[12] != v.value[12],
            u.value[13] != v.value[13],
            u.value[14] != v.value[14],
            u.value[15] != v.value[15]
            );
}

} // simd


//-------------------------------------------------------------------------------------------------
// Import SIMD intrinsics into namespace visionaray.
// Enable ADL!
//

using simd::select;
using simd::store;
using simd::any;
using simd::all;

} // MATH_NAMESPACE
