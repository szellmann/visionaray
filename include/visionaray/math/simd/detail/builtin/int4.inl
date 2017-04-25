// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <visionaray/math/detail/math.h>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// int4 members
//

MATH_FUNC
VSNRAY_FORCE_INLINE int4::basic_int(int x, int y, int z, int w)
    : value{x, y, z, w}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4::basic_int(int const v[4])
    : value{v[0], v[1], v[2], v[3]}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4::basic_int(int s)
    : value{s, s, s, s}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4::basic_int(unsigned s)
    : value{static_cast<int>(s), static_cast<int>(s), static_cast<int>(s), static_cast<int>(s)}
{
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE float4 reinterpret_as_float(int4 const& a)
{
    return *reinterpret_cast<float4 const*>(&a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE float4 convert_to_float(int4 const& a)
{
    return float4(
            static_cast<float>(a.value[0]),
            static_cast<float>(a.value[1]),
            static_cast<float>(a.value[2]),
            static_cast<float>(a.value[3])
            );
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE int4 select(mask4 const& m, int4 const& a, int4 const& b)
{
    return int4(
            m.value[0] ? a.value[0] : b.value[0],
            m.value[1] ? a.value[1] : b.value[1],
            m.value[2] ? a.value[2] : b.value[2],
            m.value[3] ? a.value[3] : b.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

MATH_FUNC
VSNRAY_FORCE_INLINE void store(int dst[4], int4 const& v)
{
    dst[0] = v.value[0];
    dst[1] = v.value[1];
    dst[2] = v.value[2];
    dst[3] = v.value[3];
}

MATH_FUNC
VSNRAY_FORCE_INLINE void store(unsigned dst[4], int4 const& v)
{
    dst[0] = static_cast<unsigned>(v.value[0]);
    dst[1] = static_cast<unsigned>(v.value[1]);
    dst[2] = static_cast<unsigned>(v.value[2]);
    dst[3] = static_cast<unsigned>(v.value[3]);
}

template <int A0, int A1, int A2, int A3>
MATH_FUNC
VSNRAY_FORCE_INLINE int4 shuffle(int4 const& a)
{
    return int4(a.value[A0], a.value[A1], a.value[A2], a.value[A3]);
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE int& get(int4& v)
{
    static_assert(I >= 0 && I < 4, "Index out of range for SIMD vector access");

    return v.value[I];
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE int const& get(int4 const& v)
{
    static_assert(I >= 0 && I < 4, "Index out of range for SIMD vector access");

    return v.value[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmethic
//

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator+(int4 const& v)
{
    return int4(
            +v.value[0],
            +v.value[1],
            +v.value[2],
            +v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator-(int4 const& v)
{
    return int4(
            -v.value[0],
            -v.value[1],
            -v.value[2],
            -v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator+(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] + v.value[0],
            u.value[1] + v.value[1],
            u.value[2] + v.value[2],
            u.value[3] + v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator-(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] + v.value[0],
            u.value[1] + v.value[1],
            u.value[2] + v.value[2],
            u.value[3] + v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator*(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] * v.value[0],
            u.value[1] * v.value[1],
            u.value[2] * v.value[2],
            u.value[3] * v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator/(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] / v.value[0],
            u.value[1] / v.value[1],
            u.value[2] / v.value[2],
            u.value[3] / v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator%(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] % v.value[0],
            u.value[1] % v.value[1],
            u.value[2] % v.value[2],
            u.value[3] % v.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator&(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] & v.value[0],
            u.value[1] & v.value[1],
            u.value[2] & v.value[2],
            u.value[3] & v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator|(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] | v.value[0],
            u.value[1] | v.value[1],
            u.value[2] | v.value[2],
            u.value[3] | v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator^(int4 const& u, int4 const& v)
{
    return int4(
            u.value[0] ^ v.value[0],
            u.value[1] ^ v.value[1],
            u.value[2] ^ v.value[2],
            u.value[3] ^ v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator<<(int4 const& a, int count)
{
    return int4(
            a.value[0] << count,
            a.value[1] << count,
            a.value[2] << count,
            a.value[3] << count
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 operator>>(int4 const& a, int count)
{
    return int4(
            a.value[0] >> count,
            a.value[1] >> count,
            a.value[2] >> count,
            a.value[3] >> count
            );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator&&(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] && v.value[0],
            u.value[1] && v.value[1],
            u.value[2] && v.value[2],
            u.value[3] && v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator||(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] || v.value[0],
            u.value[1] || v.value[1],
            u.value[2] || v.value[2],
            u.value[3] || v.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator<(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] < v.value[0],
            u.value[1] < v.value[1],
            u.value[2] < v.value[2],
            u.value[3] < v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator>(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] > v.value[0],
            u.value[1] > v.value[1],
            u.value[2] > v.value[2],
            u.value[3] > v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator==(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] == v.value[0],
            u.value[1] == v.value[1],
            u.value[2] == v.value[2],
            u.value[3] == v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator<=(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] <= v.value[0],
            u.value[1] <= v.value[1],
            u.value[2] <= v.value[2],
            u.value[3] <= v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator>=(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] >= v.value[0],
            u.value[1] >= v.value[1],
            u.value[2] >= v.value[2],
            u.value[3] >= v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask4 operator!=(int4 const& u, int4 const& v)
{
    return mask4(
            u.value[0] != v.value[0],
            u.value[1] != v.value[1],
            u.value[2] != v.value[2],
            u.value[3] != v.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

MATH_FUNC
VSNRAY_FORCE_INLINE int4 min(int4 const& u, int4 const& v)
{
    return int4(
            MATH_NAMESPACE::min(u.value[0], v.value[0]),
            MATH_NAMESPACE::min(u.value[1], v.value[1]),
            MATH_NAMESPACE::min(u.value[2], v.value[2]),
            MATH_NAMESPACE::min(u.value[3], v.value[3])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int4 max(int4 const& u, int4 const& v)
{
    return int4(
            MATH_NAMESPACE::max(u.value[0], v.value[0]),
            MATH_NAMESPACE::max(u.value[1], v.value[1]),
            MATH_NAMESPACE::max(u.value[2], v.value[2]),
            MATH_NAMESPACE::max(u.value[3], v.value[3])
            );
}

} // simd
} // MATH_NAMESPACE
