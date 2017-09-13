// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <visionaray/math/detail/math.h>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// int8 members
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8::basic_int(
        int x1,
        int x2,
        int x3,
        int x4,
        int x5,
        int x6,
        int x7,
        int x8
        )
    : value{x1, x2, x3, x4, x5, x6, x7, x8}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8::basic_int(int const v[8])
    : value{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8::basic_int(int s)
    : value{s, s, s, s, s, s, s, s}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8::basic_int(unsigned s)
    : value{
        static_cast<int>(s),
        static_cast<int>(s),
        static_cast<int>(s),
        static_cast<int>(s),
        static_cast<int>(s),
        static_cast<int>(s),
        static_cast<int>(s),
        static_cast<int>(s)
        }
{
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 reinterpret_as_float(int8 const& a)
{
    return *reinterpret_cast<float8 const*>(&a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 convert_to_float(int8 const& a)
{
    return float8(
            static_cast<float>(a.value[0]),
            static_cast<float>(a.value[1]),
            static_cast<float>(a.value[2]),
            static_cast<float>(a.value[3]),
            static_cast<float>(a.value[4]),
            static_cast<float>(a.value[5]),
            static_cast<float>(a.value[6]),
            static_cast<float>(a.value[7])
            );
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 select(mask8 const& m, int8 const& a, int8 const& b)
{
    return int8(
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
// Load / store / get
//

MATH_FUNC
VSNRAY_FORCE_INLINE void store(int dst[8], int8 const& v)
{
    dst[0] = v.value[0];
    dst[1] = v.value[1];
    dst[2] = v.value[2];
    dst[3] = v.value[3];
    dst[4] = v.value[4];
    dst[5] = v.value[5];
    dst[6] = v.value[6];
    dst[7] = v.value[7];
}

MATH_FUNC
VSNRAY_FORCE_INLINE void store(unsigned dst[8], int8 const& v)
{
    dst[0] = static_cast<unsigned>(v.value[0]);
    dst[1] = static_cast<unsigned>(v.value[1]);
    dst[2] = static_cast<unsigned>(v.value[2]);
    dst[3] = static_cast<unsigned>(v.value[3]);
    dst[4] = static_cast<unsigned>(v.value[4]);
    dst[5] = static_cast<unsigned>(v.value[5]);
    dst[6] = static_cast<unsigned>(v.value[6]);
    dst[7] = static_cast<unsigned>(v.value[7]);
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE int& get(int8& v)
{
    static_assert(I >= 0 && I < 8, "Index out of range for SIMD vector access");

    return v.value[I];
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE int const& get(int8 const& v)
{
    static_assert(I >= 0 && I < 8, "Index out of range for SIMD vector access");

    return v.value[I];
}


//-------------------------------------------------------------------------------------------------
// Basic arithmethic
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator+(int8 const& v)
{
    return int8(
            +v.value[0],
            +v.value[1],
            +v.value[2],
            +v.value[3],
            +v.value[4],
            +v.value[5],
            +v.value[6],
            +v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator-(int8 const& v)
{
    return int8(
            -v.value[0],
            -v.value[1],
            -v.value[2],
            -v.value[3],
            -v.value[4],
            -v.value[5],
            -v.value[6],
            -v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator+(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] + v.value[0],
            u.value[1] + v.value[1],
            u.value[2] + v.value[2],
            u.value[3] + v.value[3],
            u.value[4] + v.value[4],
            u.value[5] + v.value[5],
            u.value[6] + v.value[6],
            u.value[7] + v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator-(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] - v.value[0],
            u.value[1] - v.value[1],
            u.value[2] - v.value[2],
            u.value[3] - v.value[3],
            u.value[4] - v.value[4],
            u.value[5] - v.value[5],
            u.value[6] - v.value[6],
            u.value[7] - v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator*(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] * v.value[0],
            u.value[1] * v.value[1],
            u.value[2] * v.value[2],
            u.value[3] * v.value[3],
            u.value[4] * v.value[4],
            u.value[5] * v.value[5],
            u.value[6] * v.value[6],
            u.value[7] * v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator/(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] / v.value[0],
            u.value[1] / v.value[1],
            u.value[2] / v.value[2],
            u.value[3] / v.value[3],
            u.value[4] / v.value[4],
            u.value[5] / v.value[5],
            u.value[6] / v.value[6],
            u.value[7] / v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator%(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] % v.value[0],
            u.value[1] % v.value[1],
            u.value[2] % v.value[2],
            u.value[3] % v.value[3],
            u.value[4] % v.value[4],
            u.value[5] % v.value[5],
            u.value[6] % v.value[6],
            u.value[7] % v.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator&(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] & v.value[0],
            u.value[1] & v.value[1],
            u.value[2] & v.value[2],
            u.value[3] & v.value[3],
            u.value[4] & v.value[4],
            u.value[5] & v.value[5],
            u.value[6] & v.value[6],
            u.value[7] & v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator|(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] | v.value[0],
            u.value[1] | v.value[1],
            u.value[2] | v.value[2],
            u.value[3] | v.value[3],
            u.value[4] | v.value[4],
            u.value[5] | v.value[5],
            u.value[6] | v.value[6],
            u.value[7] | v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator^(int8 const& u, int8 const& v)
{
    return int8(
            u.value[0] ^ v.value[0],
            u.value[1] ^ v.value[1],
            u.value[2] ^ v.value[2],
            u.value[3] ^ v.value[3],
            u.value[4] ^ v.value[4],
            u.value[5] ^ v.value[5],
            u.value[6] ^ v.value[6],
            u.value[7] ^ v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator<<(int8 const& a, int count)
{
    return int8(
            a.value[0] << count,
            a.value[1] << count,
            a.value[2] << count,
            a.value[3] << count,
            a.value[4] << count,
            a.value[5] << count,
            a.value[6] << count,
            a.value[7] << count
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 operator>>(int8 const& a, int count)
{
    return int8(
            a.value[0] >> count,
            a.value[1] >> count,
            a.value[2] >> count,
            a.value[3] >> count,
            a.value[4] >> count,
            a.value[5] >> count,
            a.value[6] >> count,
            a.value[7] >> count
            );
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator&&(int8 const& u, int8 const& v)
{
    return mask8(
            u.value[0] && v.value[0],
            u.value[1] && v.value[1],
            u.value[2] && v.value[2],
            u.value[3] && v.value[3],
            u.value[4] && v.value[4],
            u.value[5] && v.value[5],
            u.value[6] && v.value[6],
            u.value[7] && v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator||(int8 const& u, int8 const& v)
{
    return mask8(
            u.value[0] || v.value[0],
            u.value[1] || v.value[1],
            u.value[2] || v.value[2],
            u.value[3] || v.value[3],
            u.value[4] || v.value[4],
            u.value[5] || v.value[5],
            u.value[6] || v.value[6],
            u.value[7] || v.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator<(int8 const& u, int8 const& v)
{
    return mask8(
            u.value[0] < v.value[0],
            u.value[1] < v.value[1],
            u.value[2] < v.value[2],
            u.value[3] < v.value[3],
            u.value[4] < v.value[4],
            u.value[5] < v.value[5],
            u.value[6] < v.value[6],
            u.value[7] < v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator>(int8 const& u, int8 const& v)
{
    return mask8(
            u.value[0] > v.value[0],
            u.value[1] > v.value[1],
            u.value[2] > v.value[2],
            u.value[3] > v.value[3],
            u.value[4] > v.value[4],
            u.value[5] > v.value[5],
            u.value[6] > v.value[6],
            u.value[7] > v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator==(int8 const& u, int8 const& v)
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
VSNRAY_FORCE_INLINE mask8 operator<=(int8 const& u, int8 const& v)
{
    return mask8(
            u.value[0] <= v.value[0],
            u.value[1] <= v.value[1],
            u.value[2] <= v.value[2],
            u.value[3] <= v.value[3],
            u.value[4] <= v.value[4],
            u.value[5] <= v.value[5],
            u.value[6] <= v.value[6],
            u.value[7] <= v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator>=(int8 const& u, int8 const& v)
{
    return mask8(
            u.value[0] >= v.value[0],
            u.value[1] >= v.value[1],
            u.value[2] >= v.value[2],
            u.value[3] >= v.value[3],
            u.value[4] >= v.value[4],
            u.value[5] >= v.value[5],
            u.value[6] >= v.value[6],
            u.value[7] >= v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 operator!=(int8 const& u, int8 const& v)
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


//-------------------------------------------------------------------------------------------------
// Math functions
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 min(int8 const& u, int8 const& v)
{
    return int8(
            MATH_NAMESPACE::min(u.value[0], v.value[0]),
            MATH_NAMESPACE::min(u.value[1], v.value[1]),
            MATH_NAMESPACE::min(u.value[2], v.value[2]),
            MATH_NAMESPACE::min(u.value[3], v.value[3]),
            MATH_NAMESPACE::min(u.value[4], v.value[4]),
            MATH_NAMESPACE::min(u.value[5], v.value[5]),
            MATH_NAMESPACE::min(u.value[6], v.value[6]),
            MATH_NAMESPACE::min(u.value[7], v.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE int8 max(int8 const& u, int8 const& v)
{
    return int8(
            MATH_NAMESPACE::max(u.value[0], v.value[0]),
            MATH_NAMESPACE::max(u.value[1], v.value[1]),
            MATH_NAMESPACE::max(u.value[2], v.value[2]),
            MATH_NAMESPACE::max(u.value[3], v.value[3]),
            MATH_NAMESPACE::max(u.value[4], v.value[4]),
            MATH_NAMESPACE::max(u.value[5], v.value[5]),
            MATH_NAMESPACE::max(u.value[6], v.value[6]),
            MATH_NAMESPACE::max(u.value[7], v.value[7])
            );
}

} // simd
} // MATH_NAMESPACE
