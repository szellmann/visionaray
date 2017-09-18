// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <visionaray/math/detail/math.h>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float8 members
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8::basic_float(
        float x1,
        float x2,
        float x3,
        float x4,
        float x5,
        float x6,
        float x7,
        float x8
        )
    : value{x1, x2, x3, x4, x5, x6, x7, x8}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8::basic_float(float const v[8])
    : value{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]}
{
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8::basic_float(float s)
    : value{s, s, s, s, s, s, s, s}
{
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 reinterpret_as_int(float8 const& a)
{
    return *reinterpret_cast<int8 const*>(&a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

MATH_FUNC
VSNRAY_FORCE_INLINE int8 convert_to_int(float8 const& a)
{
    return int8(
            static_cast<int>(a.value[0]),
            static_cast<int>(a.value[1]),
            static_cast<int>(a.value[2]),
            static_cast<int>(a.value[3]),
            static_cast<int>(a.value[4]),
            static_cast<int>(a.value[5]),
            static_cast<int>(a.value[6]),
            static_cast<int>(a.value[7])
            );
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 select(mask8 const& m, float8 const& a, float8 const& b)
{
    return float8(
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

//MATH_FUNC
//VSNRAY_FORCE_INLINE float8 load(float const src[8])
//{
//    return float8(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]);
//}

MATH_FUNC
VSNRAY_FORCE_INLINE void store(float dst[8], float8 const& v)
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

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE float& get(float8& v)
{
    static_assert(I < 8, "Index out of range for SIMD vector access");

    return v.value[I];
}

template <size_t I>
MATH_FUNC
VSNRAY_FORCE_INLINE float const& get(float8 const& v)
{
    static_assert(I < 8, "Index out of range for SIMD vector access");

    return v.value[I];
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 move_lo(float8 const& u, float8 const& v)
{
    return float8(
            u.value[0],
            u.value[1],
            u.value[2],
            u.value[3],
            v.value[0],
            v.value[1],
            v.value[2],
            v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 move_hi(float8 const& u, float8 const& v)
{
    return float8(
            v.value[4],
            v.value[5],
            v.value[6],
            v.value[7],
            u.value[4],
            u.value[5],
            u.value[6],
            u.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 interleave_lo(float8 const& u, float8 const& v)
{
    return float8(
            u.value[0],
            v.value[0],
            u.value[1],
            v.value[1],
            u.value[2],
            v.value[2],
            u.value[3],
            v.value[3]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 interleave_hi(float8 const& u, float8 const& v)
{
    return float8(
            u.value[4],
            v.value[4],
            u.value[5],
            v.value[5],
            u.value[6],
            v.value[6],
            u.value[7],
            v.value[7]
            );
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetics
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 operator+(float8 const& v)
{
    return float8(
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
VSNRAY_FORCE_INLINE float8 operator-(float8 const& v)
{
    return float8(
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
VSNRAY_FORCE_INLINE float8 operator+(float8 const& u, float8 const& v)
{
    return float8(
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
VSNRAY_FORCE_INLINE float8 operator-(float8 const& u, float8 const& v)
{
    return float8(
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
VSNRAY_FORCE_INLINE float8 operator*(float8 const& u, float8 const& v)
{

    return float8(
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
VSNRAY_FORCE_INLINE float8 operator/(float8 const& u, float8 const& v)
{
    return float8(
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


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 operator&(float8 const& u, float8 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[8] = {
            ui[0] & vi[0],
            ui[1] & vi[1],
            ui[2] & vi[2],
            ui[3] & vi[3],
            ui[4] & vi[4],
            ui[5] & vi[5],
            ui[6] & vi[6],
            ui[7] & vi[7]
            };

    return float8(reinterpret_cast<float*>(ri));
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 operator|(float8 const& u, float8 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[8] = {
            ui[0] | vi[0],
            ui[1] | vi[1],
            ui[2] | vi[2],
            ui[3] | vi[3],
            ui[4] | vi[4],
            ui[5] | vi[5],
            ui[6] | vi[6],
            ui[7] | vi[7]
            };

    return float8(reinterpret_cast<float*>(ri));
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 operator^(float8 const& u, float8 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[8] = {
            ui[0] ^ vi[0],
            ui[1] ^ vi[1],
            ui[2] ^ vi[2],
            ui[3] ^ vi[3],
            ui[4] ^ vi[4],
            ui[5] ^ vi[5],
            ui[6] ^ vi[6],
            ui[7] ^ vi[7]
            };

    return float8(reinterpret_cast<float*>(ri));
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 operator&&(float8 const& u, float8 const& v)
{
    return float8(
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
VSNRAY_FORCE_INLINE float8 operator||(float8 const& u, float8 const& v)
{
    return float8(
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
VSNRAY_FORCE_INLINE mask8 operator<(float8 const& u, float8 const& v)
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
VSNRAY_FORCE_INLINE mask8 operator>(float8 const& u, float8 const& v)
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
VSNRAY_FORCE_INLINE mask8 operator<=(float8 const& u, float8 const& v)
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
VSNRAY_FORCE_INLINE mask8 operator>=(float8 const& u, float8 const& v)
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
VSNRAY_FORCE_INLINE mask8 operator==(float8 const& u, float8 const& v)
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
VSNRAY_FORCE_INLINE mask8 operator!=(float8 const& u, float8 const& v)
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
VSNRAY_FORCE_INLINE float8 dot(float8 const& u, float8 const& v)
{
    return float8(
            u.value[0] * v.value[0]
          + u.value[1] * v.value[1]
          + u.value[2] * v.value[2]
          + u.value[3] * v.value[3]
          + u.value[4] * v.value[4]
          + u.value[5] * v.value[5]
          + u.value[6] * v.value[6]
          + u.value[7] * v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 min(float8 const& u, float8 const& v)
{
    return float8(
            u.value[0] < v.value[0] ? u.value[0] : v.value[0],
            u.value[1] < v.value[1] ? u.value[1] : v.value[1],
            u.value[2] < v.value[2] ? u.value[2] : v.value[2],
            u.value[3] < v.value[3] ? u.value[3] : v.value[3],
            u.value[4] < v.value[4] ? u.value[4] : v.value[4],
            u.value[5] < v.value[5] ? u.value[5] : v.value[5],
            u.value[6] < v.value[6] ? u.value[6] : v.value[6],
            u.value[7] < v.value[7] ? u.value[7] : v.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 max(float8 const& u, float8 const& v)
{
    return float8(
            u.value[0] < v.value[0] ? v.value[0] : u.value[0],
            u.value[1] < v.value[1] ? v.value[1] : u.value[1],
            u.value[2] < v.value[2] ? v.value[2] : u.value[2],
            u.value[3] < v.value[3] ? v.value[3] : u.value[3],
            u.value[4] < v.value[4] ? v.value[4] : u.value[4],
            u.value[5] < v.value[5] ? v.value[5] : u.value[5],
            u.value[6] < v.value[6] ? v.value[6] : u.value[6],
            u.value[7] < v.value[7] ? v.value[7] : u.value[7]
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 saturate(float8 const& u)
{
    return max(float8(0.0f), min(u, float8(1.0f)));
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 abs(float8 const& u)
{
    return float8(
            fabsf(u.value[0]),
            fabsf(u.value[1]),
            fabsf(u.value[2]),
            fabsf(u.value[3]),
            fabsf(u.value[4]),
            fabsf(u.value[5]),
            fabsf(u.value[6]),
            fabsf(u.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 round(float8 const& v)
{
    return float8(
            roundf(v.value[0]),
            roundf(v.value[1]),
            roundf(v.value[2]),
            roundf(v.value[3]),
            roundf(v.value[4]),
            roundf(v.value[5]),
            roundf(v.value[6]),
            roundf(v.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 ceil(float8 const& v)
{
    return float8(
            ceilf(v.value[0]),
            ceilf(v.value[1]),
            ceilf(v.value[2]),
            ceilf(v.value[3]),
            ceilf(v.value[4]),
            ceilf(v.value[5]),
            ceilf(v.value[6]),
            ceilf(v.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 floor(float8 const& v)
{
    return float8(
            floorf(v.value[0]),
            floorf(v.value[1]),
            floorf(v.value[2]),
            floorf(v.value[3]),
            floorf(v.value[4]),
            floorf(v.value[5]),
            floorf(v.value[6]),
            floorf(v.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 sqrt(float8 const& v)
{
    return float8(
            sqrtf(v.value[0]),
            sqrtf(v.value[1]),
            sqrtf(v.value[2]),
            sqrtf(v.value[3]),
            sqrtf(v.value[4]),
            sqrtf(v.value[5]),
            sqrtf(v.value[6]),
            sqrtf(v.value[7])
            );
}

#if !defined(__KALMAR_ACCELERATOR__)
MATH_FUNC
VSNRAY_FORCE_INLINE mask8 isinf(float8 const& v)
{
    return mask8(
            MATH_NAMESPACE::isinf(v.value[0]),
            MATH_NAMESPACE::isinf(v.value[1]),
            MATH_NAMESPACE::isinf(v.value[2]),
            MATH_NAMESPACE::isinf(v.value[3]),
            MATH_NAMESPACE::isinf(v.value[4]),
            MATH_NAMESPACE::isinf(v.value[5]),
            MATH_NAMESPACE::isinf(v.value[6]),
            MATH_NAMESPACE::isinf(v.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 isnan(float8 const& v)
{
    return mask8(
            MATH_NAMESPACE::isnan(v.value[0]),
            MATH_NAMESPACE::isnan(v.value[1]),
            MATH_NAMESPACE::isnan(v.value[2]),
            MATH_NAMESPACE::isnan(v.value[3]),
            MATH_NAMESPACE::isnan(v.value[4]),
            MATH_NAMESPACE::isnan(v.value[5]),
            MATH_NAMESPACE::isnan(v.value[6]),
            MATH_NAMESPACE::isnan(v.value[7])
            );
}

MATH_FUNC
VSNRAY_FORCE_INLINE mask8 isfinite(float8 const& v)
{
    return mask8(
            MATH_NAMESPACE::isfinite(v.value[0]),
            MATH_NAMESPACE::isfinite(v.value[1]),
            MATH_NAMESPACE::isfinite(v.value[2]),
            MATH_NAMESPACE::isfinite(v.value[3]),
            MATH_NAMESPACE::isfinite(v.value[4]),
            MATH_NAMESPACE::isfinite(v.value[5]),
            MATH_NAMESPACE::isfinite(v.value[6]),
            MATH_NAMESPACE::isfinite(v.value[7])
            );
}
#endif


//-------------------------------------------------------------------------------------------------
//
//

MATH_FUNC
VSNRAY_FORCE_INLINE float8 rcp(float8 const& v)
{
    return float8(1.0f) / v;
}

MATH_FUNC
VSNRAY_FORCE_INLINE float8 rsqrt(float8 const& v)
{
    return float8(1.0f) / sqrt(v);
}

} // simd
} // MATH_NAMESPACE
