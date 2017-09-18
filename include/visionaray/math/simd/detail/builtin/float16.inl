// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <visionaray/math/detail/math.h>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float16 members
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16::basic_float(
        float  x1, float  x2, float  x3, float  x4,
        float  x5, float  x6, float  x7, float  x8,
        float  x9, float x10, float x11, float x12,
        float x13, float x14, float x15, float x16
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
VSNRAY_FORCE_INLINE float16::basic_float(float const v[16])
    : value{
        v[ 0], v[ 1], v[ 2], v[ 3],
        v[ 4], v[ 5], v[ 6], v[ 7],
        v[ 8], v[ 9], v[10], v[11],
        v[12], v[13], v[14], v[15]
        }
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16::basic_float(float s)
    : value{s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s}
{
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int16 reinterpret_as_int(float16 const& a)
{
    return *reinterpret_cast<int16 const*>(&a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int16 convert_to_int(float16 const& a)
{
    return int16(
            static_cast<int>(a.value[ 0]),
            static_cast<int>(a.value[ 1]),
            static_cast<int>(a.value[ 2]),
            static_cast<int>(a.value[ 3]),
            static_cast<int>(a.value[ 4]),
            static_cast<int>(a.value[ 5]),
            static_cast<int>(a.value[ 6]),
            static_cast<int>(a.value[ 7]),
            static_cast<int>(a.value[ 8]),
            static_cast<int>(a.value[ 9]),
            static_cast<int>(a.value[10]),
            static_cast<int>(a.value[11]),
            static_cast<int>(a.value[12]),
            static_cast<int>(a.value[13]),
            static_cast<int>(a.value[14]),
            static_cast<int>(a.value[15])
            );
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16 select(mask16 const& m, float16 const& a, float16 const& b)
{
    return float16(
            m.value[ 0] ? a.value[ 0] : b.value[ 0],
            m.value[ 1] ? a.value[ 1] : b.value[ 1],
            m.value[ 2] ? a.value[ 2] : b.value[ 2],
            m.value[ 3] ? a.value[ 3] : b.value[ 3],
            m.value[ 4] ? a.value[ 4] : b.value[ 4],
            m.value[ 5] ? a.value[ 5] : b.value[ 5],
            m.value[ 6] ? a.value[ 6] : b.value[ 6],
            m.value[ 7] ? a.value[ 7] : b.value[ 7],
            m.value[ 8] ? a.value[ 0] : b.value[ 8],
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
// Load / store / get
//

//MATH_FUNC
//VSNRAY_FORCE_INLINE float16 load(float const src[16])
//{
//    return float16(
//        src[ 0], src[ 1], src[ 2], src[ 3],
//        src[ 4], src[ 5], src[ 6], src[ 7],
//        src[ 8], src[ 9], src[10], src[11],
//        src[12], src[13], src[14], src[15]
//        );
//}

MATH_FUNC
VSNRAY_FORCE_INLINE void store(float dst[16], float16 const& v)
{
    dst[ 0] = v.value[ 0];
    dst[ 1] = v.value[ 1];
    dst[ 2] = v.value[ 2];
    dst[ 3] = v.value[ 3];
    dst[ 4] = v.value[ 4];
    dst[ 5] = v.value[ 5];
    dst[ 6] = v.value[ 6];
    dst[ 7] = v.value[ 7];
    dst[ 8] = v.value[ 8];
    dst[ 9] = v.value[ 9];
    dst[10] = v.value[10];
    dst[11] = v.value[11];
    dst[12] = v.value[12];
    dst[13] = v.value[13];
    dst[14] = v.value[14];
    dst[15] = v.value[15];
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE float& get(float16& v)
{
    static_assert(I < 16, "Index out of range for SIMD vector access");

    return v.value[I];
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE float const& get(float16 const& v)
{
    static_assert(I < 16, "Index out of range for SIMD vector access");

    return v.value[I];
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 move_lo(float16 const& u, float16 const& v)
{
    return float16(
            u.value[0],
            u.value[1],
            u.value[2],
            u.value[3],
            u.value[4],
            u.value[5],
            u.value[6],
            u.value[7],
            v.value[0],
            v.value[1],
            v.value[2],
            v.value[3],
            v.value[4],
            v.value[5],
            v.value[6],
            v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 move_hi(float16 const& u, float16 const& v)
{
    return float16(
            v.value[ 8],
            v.value[ 9],
            v.value[10],
            v.value[11],
            v.value[12],
            v.value[13],
            v.value[14],
            v.value[15],
            u.value[ 8],
            u.value[ 9],
            u.value[10],
            u.value[11],
            u.value[12],
            u.value[13],
            u.value[14],
            u.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 interleave_lo(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0],
            v.value[ 0],
            u.value[ 1],
            v.value[ 1],
            u.value[ 2],
            v.value[ 2],
            u.value[ 3],
            v.value[ 3],
            u.value[ 4],
            v.value[ 4],
            u.value[ 5],
            v.value[ 5],
            u.value[ 6],
            v.value[ 6],
            u.value[ 7],
            v.value[ 7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 interleave_hi(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 8],
            v.value[ 8],
            u.value[ 9],
            v.value[ 9],
            u.value[10],
            v.value[10],
            u.value[11],
            v.value[11],
            u.value[12],
            v.value[12],
            u.value[13],
            v.value[13],
            u.value[14],
            v.value[14],
            u.value[15],
            v.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetics
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator+(float16 const& v)
{
    return float16(
            +v.value[ 0],
            +v.value[ 1],
            +v.value[ 2],
            +v.value[ 3],
            +v.value[ 4],
            +v.value[ 5],
            +v.value[ 6],
            +v.value[ 7],
            +v.value[ 8],
            +v.value[ 9],
            +v.value[10],
            +v.value[11],
            +v.value[12],
            +v.value[13],
            +v.value[14],
            +v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator-(float16 const& v)
{
    return float16(
            -v.value[ 0],
            -v.value[ 1],
            -v.value[ 2],
            -v.value[ 3],
            -v.value[ 4],
            -v.value[ 5],
            -v.value[ 6],
            -v.value[ 7],
            -v.value[ 8],
            -v.value[ 9],
            -v.value[10],
            -v.value[11],
            -v.value[12],
            -v.value[13],
            -v.value[14],
            -v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator+(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] + v.value[ 0],
            u.value[ 1] + v.value[ 1],
            u.value[ 2] + v.value[ 2],
            u.value[ 3] + v.value[ 3],
            u.value[ 4] + v.value[ 4],
            u.value[ 5] + v.value[ 5],
            u.value[ 6] + v.value[ 6],
            u.value[ 7] + v.value[ 7],
            u.value[ 8] + v.value[ 8],
            u.value[ 9] + v.value[ 9],
            u.value[10] + v.value[10],
            u.value[11] + v.value[11],
            u.value[12] + v.value[12],
            u.value[13] + v.value[13],
            u.value[14] + v.value[14],
            u.value[15] + v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator-(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] - v.value[ 0],
            u.value[ 1] - v.value[ 1],
            u.value[ 2] - v.value[ 2],
            u.value[ 3] - v.value[ 3],
            u.value[ 4] - v.value[ 4],
            u.value[ 5] - v.value[ 5],
            u.value[ 6] - v.value[ 6],
            u.value[ 7] - v.value[ 7],
            u.value[ 8] - v.value[ 8],
            u.value[ 9] - v.value[ 9],
            u.value[10] - v.value[10],
            u.value[11] - v.value[11],
            u.value[12] - v.value[12],
            u.value[13] - v.value[13],
            u.value[14] - v.value[14],
            u.value[15] - v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator*(float16 const& u, float16 const& v)
{

    return float16(
            u.value[ 0] * v.value[ 0],
            u.value[ 1] * v.value[ 1],
            u.value[ 2] * v.value[ 2],
            u.value[ 3] * v.value[ 3],
            u.value[ 4] * v.value[ 4],
            u.value[ 5] * v.value[ 5],
            u.value[ 6] * v.value[ 6],
            u.value[ 7] * v.value[ 7],
            u.value[ 8] * v.value[ 8],
            u.value[ 9] * v.value[ 9],
            u.value[10] * v.value[10],
            u.value[11] * v.value[11],
            u.value[12] * v.value[12],
            u.value[13] * v.value[13],
            u.value[14] * v.value[14],
            u.value[15] * v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator/(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] / v.value[ 0],
            u.value[ 1] / v.value[ 1],
            u.value[ 2] / v.value[ 2],
            u.value[ 3] / v.value[ 3],
            u.value[ 4] / v.value[ 4],
            u.value[ 5] / v.value[ 5],
            u.value[ 6] / v.value[ 6],
            u.value[ 7] / v.value[ 7],
            u.value[ 8] / v.value[ 8],
            u.value[ 9] / v.value[ 9],
            u.value[10] / v.value[10],
            u.value[11] / v.value[11],
            u.value[12] / v.value[12],
            u.value[13] / v.value[13],
            u.value[14] / v.value[14],
            u.value[15] / v.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator&(float16 const& u, float16 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[16] = {
            ui[ 0] & vi[ 0],
            ui[ 1] & vi[ 1],
            ui[ 2] & vi[ 2],
            ui[ 3] & vi[ 3],
            ui[ 4] & vi[ 4],
            ui[ 5] & vi[ 5],
            ui[ 6] & vi[ 6],
            ui[ 7] & vi[ 7],
            ui[ 8] & vi[ 8],
            ui[ 9] & vi[ 9],
            ui[10] & vi[10],
            ui[11] & vi[11],
            ui[12] & vi[12],
            ui[13] & vi[13],
            ui[14] & vi[14],
            ui[15] & vi[15]
            };

    return float16(reinterpret_cast<float*>(ri));
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator|(float16 const& u, float16 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[16] = {
            ui[ 0] | vi[ 0],
            ui[ 1] | vi[ 1],
            ui[ 2] | vi[ 2],
            ui[ 3] | vi[ 3],
            ui[ 4] | vi[ 4],
            ui[ 5] | vi[ 5],
            ui[ 6] | vi[ 6],
            ui[ 7] | vi[ 7],
            ui[ 8] | vi[ 8],
            ui[ 9] | vi[ 9],
            ui[10] | vi[10],
            ui[11] | vi[11],
            ui[12] | vi[12],
            ui[13] | vi[13],
            ui[14] | vi[14],
            ui[15] | vi[15]
            };

    return float16(reinterpret_cast<float*>(ri));
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator^(float16 const& u, float16 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[16] = {
            ui[ 0] ^ vi[ 0],
            ui[ 1] ^ vi[ 1],
            ui[ 2] ^ vi[ 2],
            ui[ 3] ^ vi[ 3],
            ui[ 4] ^ vi[ 4],
            ui[ 5] ^ vi[ 5],
            ui[ 6] ^ vi[ 6],
            ui[ 7] ^ vi[ 7],
            ui[ 8] ^ vi[ 8],
            ui[ 9] ^ vi[ 9],
            ui[10] ^ vi[10],
            ui[11] ^ vi[11],
            ui[12] ^ vi[12],
            ui[13] ^ vi[13],
            ui[14] ^ vi[14],
            ui[15] ^ vi[15]
            };

    return float16(reinterpret_cast<float*>(ri));
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator&&(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] && v.value[ 0],
            u.value[ 1] && v.value[ 1],
            u.value[ 2] && v.value[ 2],
            u.value[ 3] && v.value[ 3],
            u.value[ 4] && v.value[ 4],
            u.value[ 5] && v.value[ 5],
            u.value[ 6] && v.value[ 6],
            u.value[ 7] && v.value[ 7],
            u.value[ 8] && v.value[ 8],
            u.value[ 9] && v.value[ 9],
            u.value[10] && v.value[10],
            u.value[11] && v.value[11],
            u.value[12] && v.value[12],
            u.value[13] && v.value[13],
            u.value[14] && v.value[14],
            u.value[15] && v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 operator||(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] || v.value[ 0],
            u.value[ 1] || v.value[ 1],
            u.value[ 2] || v.value[ 2],
            u.value[ 3] || v.value[ 3],
            u.value[ 4] || v.value[ 4],
            u.value[ 5] || v.value[ 5],
            u.value[ 6] || v.value[ 6],
            u.value[ 7] || v.value[ 7],
            u.value[ 8] || v.value[ 8],
            u.value[ 9] || v.value[ 9],
            u.value[10] || v.value[10],
            u.value[11] || v.value[11],
            u.value[12] || v.value[12],
            u.value[13] || v.value[13],
            u.value[14] || v.value[14],
            u.value[15] || v.value[15]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator<(float16 const& u, float16 const& v)
{
    return mask16(
            u.value[ 0] < v.value[ 0],
            u.value[ 1] < v.value[ 1],
            u.value[ 2] < v.value[ 2],
            u.value[ 3] < v.value[ 3],
            u.value[ 4] < v.value[ 4],
            u.value[ 5] < v.value[ 5],
            u.value[ 6] < v.value[ 6],
            u.value[ 7] < v.value[ 7],
            u.value[ 8] < v.value[ 8],
            u.value[ 9] < v.value[ 9],
            u.value[10] < v.value[10],
            u.value[11] < v.value[11],
            u.value[12] < v.value[12],
            u.value[13] < v.value[13],
            u.value[14] < v.value[14],
            u.value[15] < v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator>(float16 const& u, float16 const& v)
{
    return mask16(
            u.value[ 0] > v.value[ 0],
            u.value[ 1] > v.value[ 1],
            u.value[ 2] > v.value[ 2],
            u.value[ 3] > v.value[ 3],
            u.value[ 4] > v.value[ 4],
            u.value[ 5] > v.value[ 5],
            u.value[ 6] > v.value[ 6],
            u.value[ 7] > v.value[ 7],
            u.value[ 8] > v.value[ 8],
            u.value[ 9] > v.value[ 9],
            u.value[10] > v.value[10],
            u.value[11] > v.value[11],
            u.value[12] > v.value[12],
            u.value[13] > v.value[13],
            u.value[14] > v.value[14],
            u.value[15] > v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator<=(float16 const& u, float16 const& v)
{
    return mask16(
            u.value[ 0] <= v.value[ 0],
            u.value[ 1] <= v.value[ 1],
            u.value[ 2] <= v.value[ 2],
            u.value[ 3] <= v.value[ 3],
            u.value[ 4] <= v.value[ 4],
            u.value[ 5] <= v.value[ 5],
            u.value[ 6] <= v.value[ 6],
            u.value[ 7] <= v.value[ 7],
            u.value[ 8] <= v.value[ 8],
            u.value[ 9] <= v.value[ 9],
            u.value[10] <= v.value[10],
            u.value[11] <= v.value[11],
            u.value[12] <= v.value[12],
            u.value[13] <= v.value[13],
            u.value[14] <= v.value[14],
            u.value[15] <= v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator>=(float16 const& u, float16 const& v)
{
    return mask16(
            u.value[ 0] >= v.value[ 0],
            u.value[ 1] >= v.value[ 1],
            u.value[ 2] >= v.value[ 2],
            u.value[ 3] >= v.value[ 3],
            u.value[ 4] >= v.value[ 4],
            u.value[ 5] >= v.value[ 5],
            u.value[ 6] >= v.value[ 6],
            u.value[ 7] >= v.value[ 7],
            u.value[ 8] >= v.value[ 8],
            u.value[ 9] >= v.value[ 9],
            u.value[10] >= v.value[10],
            u.value[11] >= v.value[11],
            u.value[12] >= v.value[12],
            u.value[13] >= v.value[13],
            u.value[14] >= v.value[14],
            u.value[15] >= v.value[15]
            );
} 

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 operator==(float16 const& u, float16 const& v)
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
VSNRAY_FORCE_INLINE mask16 operator!=(float16 const& u, float16 const& v)
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


//-------------------------------------------------------------------------------------------------
// Math functions
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16 dot(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] * v.value[ 0]
          + u.value[ 1] * v.value[ 1]
          + u.value[ 2] * v.value[ 2]
          + u.value[ 3] * v.value[ 3]
          + u.value[ 4] * v.value[ 4]
          + u.value[ 5] * v.value[ 5]
          + u.value[ 6] * v.value[ 6]
          + u.value[ 7] * v.value[ 7]
          + u.value[ 8] * v.value[ 8]
          + u.value[ 9] * v.value[ 9]
          + u.value[10] * v.value[10]
          + u.value[11] * v.value[11]
          + u.value[12] * v.value[12]
          + u.value[13] * v.value[13]
          + u.value[14] * v.value[14]
          + u.value[15] * v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 min(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] < v.value[ 0] ? u.value[ 0] : v.value[ 0],
            u.value[ 1] < v.value[ 1] ? u.value[ 1] : v.value[ 1],
            u.value[ 2] < v.value[ 2] ? u.value[ 2] : v.value[ 2],
            u.value[ 3] < v.value[ 3] ? u.value[ 3] : v.value[ 3],
            u.value[ 4] < v.value[ 4] ? u.value[ 4] : v.value[ 4],
            u.value[ 5] < v.value[ 5] ? u.value[ 5] : v.value[ 5],
            u.value[ 6] < v.value[ 6] ? u.value[ 6] : v.value[ 6],
            u.value[ 7] < v.value[ 7] ? u.value[ 7] : v.value[ 7],
            u.value[ 8] < v.value[ 9] ? u.value[ 8] : v.value[ 8],
            u.value[ 9] < v.value[ 9] ? u.value[ 9] : v.value[ 9],
            u.value[10] < v.value[10] ? u.value[10] : v.value[10],
            u.value[11] < v.value[11] ? u.value[11] : v.value[11],
            u.value[12] < v.value[12] ? u.value[12] : v.value[12],
            u.value[13] < v.value[13] ? u.value[13] : v.value[13],
            u.value[14] < v.value[14] ? u.value[14] : v.value[14],
            u.value[15] < v.value[15] ? u.value[15] : v.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 max(float16 const& u, float16 const& v)
{
    return float16(
            u.value[ 0] < v.value[ 0] ? v.value[ 0] : u.value[ 0],
            u.value[ 1] < v.value[ 1] ? v.value[ 1] : u.value[ 1],
            u.value[ 2] < v.value[ 2] ? v.value[ 2] : u.value[ 2],
            u.value[ 3] < v.value[ 3] ? v.value[ 3] : u.value[ 3],
            u.value[ 4] < v.value[ 4] ? v.value[ 4] : u.value[ 4],
            u.value[ 5] < v.value[ 5] ? v.value[ 5] : u.value[ 5],
            u.value[ 6] < v.value[ 6] ? v.value[ 6] : u.value[ 6],
            u.value[ 7] < v.value[ 7] ? v.value[ 7] : u.value[ 7],
            u.value[ 8] < v.value[ 9] ? v.value[ 8] : u.value[ 8],
            u.value[ 9] < v.value[ 9] ? v.value[ 9] : u.value[ 9],
            u.value[10] < v.value[10] ? v.value[10] : u.value[10],
            u.value[11] < v.value[11] ? v.value[11] : u.value[11],
            u.value[12] < v.value[12] ? v.value[12] : u.value[12],
            u.value[13] < v.value[13] ? v.value[13] : u.value[13],
            u.value[14] < v.value[14] ? v.value[14] : u.value[14],
            u.value[15] < v.value[15] ? v.value[15] : u.value[15]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 saturate(float16 const& u)
{
    return max(float16(0.0f), min(u, float16(1.0f)));
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 abs(float16 const& u)
{
    return float16(
            fabsf(u.value[ 0]),
            fabsf(u.value[ 1]),
            fabsf(u.value[ 2]),
            fabsf(u.value[ 3]),
            fabsf(u.value[ 4]),
            fabsf(u.value[ 5]),
            fabsf(u.value[ 6]),
            fabsf(u.value[ 7]),
            fabsf(u.value[ 8]),
            fabsf(u.value[ 9]),
            fabsf(u.value[10]),
            fabsf(u.value[11]),
            fabsf(u.value[12]),
            fabsf(u.value[13]),
            fabsf(u.value[14]),
            fabsf(u.value[15])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 round(float16 const& v)
{
    return float16(
            roundf(v.value[ 0]),
            roundf(v.value[ 1]),
            roundf(v.value[ 2]),
            roundf(v.value[ 3]),
            roundf(v.value[ 4]),
            roundf(v.value[ 5]),
            roundf(v.value[ 6]),
            roundf(v.value[ 7]),
            roundf(v.value[ 8]),
            roundf(v.value[ 9]),
            roundf(v.value[10]),
            roundf(v.value[11]),
            roundf(v.value[12]),
            roundf(v.value[13]),
            roundf(v.value[14]),
            roundf(v.value[15])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 ceil(float16 const& v)
{
    return float16(
            ceilf(v.value[ 0]),
            ceilf(v.value[ 1]),
            ceilf(v.value[ 2]),
            ceilf(v.value[ 3]),
            ceilf(v.value[ 4]),
            ceilf(v.value[ 5]),
            ceilf(v.value[ 6]),
            ceilf(v.value[ 7]),
            ceilf(v.value[ 8]),
            ceilf(v.value[ 9]),
            ceilf(v.value[10]),
            ceilf(v.value[11]),
            ceilf(v.value[12]),
            ceilf(v.value[13]),
            ceilf(v.value[14]),
            ceilf(v.value[15])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 floor(float16 const& v)
{
    return float16(
            floorf(v.value[ 0]),
            floorf(v.value[ 1]),
            floorf(v.value[ 2]),
            floorf(v.value[ 3]),
            floorf(v.value[ 4]),
            floorf(v.value[ 5]),
            floorf(v.value[ 6]),
            floorf(v.value[ 7]),
            floorf(v.value[ 8]),
            floorf(v.value[ 9]),
            floorf(v.value[10]),
            floorf(v.value[11]),
            floorf(v.value[12]),
            floorf(v.value[13]),
            floorf(v.value[14]),
            floorf(v.value[15])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 sqrt(float16 const& v)
{
    return float16(
            sqrtf(v.value[ 0]),
            sqrtf(v.value[ 1]),
            sqrtf(v.value[ 2]),
            sqrtf(v.value[ 3]),
            sqrtf(v.value[ 4]),
            sqrtf(v.value[ 5]),
            sqrtf(v.value[ 6]),
            sqrtf(v.value[ 7]),
            sqrtf(v.value[ 8]),
            sqrtf(v.value[ 9]),
            sqrtf(v.value[10]),
            sqrtf(v.value[11]),
            sqrtf(v.value[12]),
            sqrtf(v.value[13]),
            sqrtf(v.value[14]),
            sqrtf(v.value[15])
            );
}

#if !defined(__KALMAR_ACCELERATOR__)
MATH_FUNC
VSNRAY_FORCE_INLINE mask16 isinf(float16 const& v)
{
    return mask16(
            MATH_NAMESPACE::isinf(v.value[ 0]),
            MATH_NAMESPACE::isinf(v.value[ 1]),
            MATH_NAMESPACE::isinf(v.value[ 2]),
            MATH_NAMESPACE::isinf(v.value[ 3]),
            MATH_NAMESPACE::isinf(v.value[ 4]),
            MATH_NAMESPACE::isinf(v.value[ 5]),
            MATH_NAMESPACE::isinf(v.value[ 6]),
            MATH_NAMESPACE::isinf(v.value[ 7]),
            MATH_NAMESPACE::isinf(v.value[ 8]),
            MATH_NAMESPACE::isinf(v.value[ 9]),
            MATH_NAMESPACE::isinf(v.value[10]),
            MATH_NAMESPACE::isinf(v.value[11]),
            MATH_NAMESPACE::isinf(v.value[12]),
            MATH_NAMESPACE::isinf(v.value[13]),
            MATH_NAMESPACE::isinf(v.value[14]),
            MATH_NAMESPACE::isinf(v.value[15])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 isnan(float16 const& v)
{
    return mask16(
            MATH_NAMESPACE::isnan(v.value[ 0]),
            MATH_NAMESPACE::isnan(v.value[ 1]),
            MATH_NAMESPACE::isnan(v.value[ 2]),
            MATH_NAMESPACE::isnan(v.value[ 3]),
            MATH_NAMESPACE::isnan(v.value[ 4]),
            MATH_NAMESPACE::isnan(v.value[ 5]),
            MATH_NAMESPACE::isnan(v.value[ 6]),
            MATH_NAMESPACE::isnan(v.value[ 7]),
            MATH_NAMESPACE::isnan(v.value[ 8]),
            MATH_NAMESPACE::isnan(v.value[ 9]),
            MATH_NAMESPACE::isnan(v.value[10]),
            MATH_NAMESPACE::isnan(v.value[11]),
            MATH_NAMESPACE::isnan(v.value[12]),
            MATH_NAMESPACE::isnan(v.value[13]),
            MATH_NAMESPACE::isnan(v.value[14]),
            MATH_NAMESPACE::isnan(v.value[15])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask16 isfinite(float16 const& v)
{
    return mask16(
            MATH_NAMESPACE::isfinite(v.value[ 0]),
            MATH_NAMESPACE::isfinite(v.value[ 1]),
            MATH_NAMESPACE::isfinite(v.value[ 2]),
            MATH_NAMESPACE::isfinite(v.value[ 3]),
            MATH_NAMESPACE::isfinite(v.value[ 4]),
            MATH_NAMESPACE::isfinite(v.value[ 5]),
            MATH_NAMESPACE::isfinite(v.value[ 6]),
            MATH_NAMESPACE::isfinite(v.value[ 7]),
            MATH_NAMESPACE::isfinite(v.value[ 8]),
            MATH_NAMESPACE::isfinite(v.value[ 9]),
            MATH_NAMESPACE::isfinite(v.value[10]),
            MATH_NAMESPACE::isfinite(v.value[11]),
            MATH_NAMESPACE::isfinite(v.value[12]),
            MATH_NAMESPACE::isfinite(v.value[13]),
            MATH_NAMESPACE::isfinite(v.value[14]),
            MATH_NAMESPACE::isfinite(v.value[15])
            );
}
#endif


//-------------------------------------------------------------------------------------------------
//
//

MATH_FUNC
VSNRAY_FORCE_INLINE float16 rcp(float16 const& v)
{
    return float16(1.0f) / v;
}

MATH_FUNC
VSNRAY_FORCE_INLINE float16 rsqrt(float16 const& v)
{
    return float16(1.0f) / sqrt(v);
}

} // simd
} // MATH_NAMESPACE
