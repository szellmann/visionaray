// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cmath>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float4 members
//

VSNRAY_FORCE_INLINE float4::basic_float(float x, float y, float z, float w)
    : value{x, y, z, w}
{
}

VSNRAY_FORCE_INLINE float4::basic_float(float const v[4])
    : value{v[0], v[1], v[2], v[3]}
{
}

VSNRAY_FORCE_INLINE float4::basic_float(float s)
    : value{s, s, s, s}
{
}


//-------------------------------------------------------------------------------------------------
// Bitwise cast
//

VSNRAY_FORCE_INLINE int4 reinterpret_as_int(float4 const& a)
{
    return *reinterpret_cast<int4 const*>(&a);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int4 convert_to_int(float4 const& a)
{
    return int4(
            static_cast<int>(a.value[0]),
            static_cast<int>(a.value[1]),
            static_cast<int>(a.value[2]),
            static_cast<int>(a.value[3])
            );
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE float4 select(mask4 const& m, float4 const& a, float4 const& b)
{
    return float4(
            m.value[0] ? a.value[0] : b.value[0],
            m.value[1] ? a.value[1] : b.value[1],
            m.value[2] ? a.value[2] : b.value[2],
            m.value[3] ? a.value[3] : b.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Load / store / get
//

VSNRAY_FORCE_INLINE float4 load(float const src[4])
{
    return float4(src[0], src[1], src[2], src[3]);
}

VSNRAY_FORCE_INLINE void store(float dst[4], float4 const& v)
{
    dst[0] = v.value[0];
    dst[1] = v.value[1];
    dst[2] = v.value[2];
    dst[3] = v.value[3];
}

template <size_t I>
VSNRAY_FORCE_INLINE float& get(float4& v)
{
    static_assert(I < 4, "Index out of range for SIMD vector access");

    return v.value[I];
}

template <size_t I>
VSNRAY_FORCE_INLINE float const& get(float4 const& v)
{
    static_assert(I < 4, "Index out of range for SIMD vector access");

    return v.value[I];
}

template <int U0, int U1, int V2, int V3>
VSNRAY_FORCE_INLINE float4 shuffle(float4 const& u, float4 const& v)
{
    return float4(u.value[U0], u.value[U1], v.value[V2], v.value[V3]);
}

template <int V0, int V1, int V2, int V3>
VSNRAY_FORCE_INLINE float4 shuffle(float4 const& v)
{
    return float4(v.value[V0], v.value[V1], v.value[V2], v.value[V3]);
}

VSNRAY_FORCE_INLINE float4 move_lo(float4 const& u, float4 const& v)
{
    return float4(u.value[0], u.value[1], v.value[0], v.value[1]);
}

VSNRAY_FORCE_INLINE float4 move_hi(float4 const& u, float4 const& v)
{
    return float4(v.value[2], v.value[3], u.value[2], u.value[3]);
}

VSNRAY_FORCE_INLINE float4 interleave_lo(float4 const& u, float4 const& v)
{
    return float4(u.value[0], v.value[0], u.value[1], v.value[1]);
}

VSNRAY_FORCE_INLINE float4 interleave_hi(float4 const& u, float4 const& v)
{
    return float4(u.value[2], v.value[2], u.value[3], v.value[3]);
}


//-------------------------------------------------------------------------------------------------
// Basic arithmetics
//

VSNRAY_FORCE_INLINE float4 operator+(float4 const& v)
{
    return float4(
            +v.value[0],
            +v.value[1],
            +v.value[2],
            +v.value[3]
            );
}

VSNRAY_FORCE_INLINE float4 operator-(float4 const& v)
{
    return float4(
            -v.value[0],
            -v.value[1],
            -v.value[2],
            -v.value[3]
            );
}

VSNRAY_FORCE_INLINE float4 operator+(float4 const& u, float4 const& v)
{
    return float4(
            u.value[0] + v.value[0],
            u.value[1] + v.value[1],
            u.value[2] + v.value[2],
            u.value[3] + v.value[3]
            );
}

VSNRAY_FORCE_INLINE float4 operator-(float4 const& u, float4 const& v)
{
    return float4(
            u.value[0] - v.value[0],
            u.value[1] - v.value[1],
            u.value[2] - v.value[2],
            u.value[3] - v.value[3]
            );
}

VSNRAY_FORCE_INLINE float4 operator*(float4 const& u, float4 const& v)
{

    return float4(
            u.value[0] * v.value[0],
            u.value[1] * v.value[1],
            u.value[2] * v.value[2],
            u.value[3] * v.value[3]
            );
}

VSNRAY_FORCE_INLINE float4 operator/(float4 const& u, float4 const& v)
{
    return float4(
            u.value[0] / v.value[0],
            u.value[1] / v.value[1],
            u.value[2] / v.value[2],
            u.value[3] / v.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE float4 operator&(float4 const& u, float4 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[4] = { ui[0] & vi[0], ui[1] & vi[1], ui[2] & vi[2], ui[3] & vi[3] };

    return float4(reinterpret_cast<float*>(ri));
}

VSNRAY_FORCE_INLINE float4 operator|(float4 const& u, float4 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[4] = { ui[0] | vi[0], ui[1] | vi[1], ui[2] | vi[2], ui[3] | vi[3] };

    return float4(reinterpret_cast<float*>(ri));
}

VSNRAY_FORCE_INLINE float4 operator^(float4 const& u, float4 const& v)
{
    int const* ui = reinterpret_cast<int const*>(u.value);
    int const* vi = reinterpret_cast<int const*>(v.value);

    int ri[4] = { ui[0] ^ vi[0], ui[1] ^ vi[1], ui[2] ^ vi[2], ui[3] ^ vi[3] };

    return float4(reinterpret_cast<float*>(ri));
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE float4 operator&&(float4 const& u, float4 const& v)
{
    return float4(
            u.value[0] && v.value[0],
            u.value[1] && v.value[1],
            u.value[2] && v.value[2],
            u.value[3] && v.value[3]
            );
}

VSNRAY_FORCE_INLINE float4 operator||(float4 const& u, float4 const& v)
{
    return float4(
            u.value[0] || v.value[0],
            u.value[1] || v.value[1],
            u.value[2] || v.value[2],
            u.value[3] || v.value[3]
            );
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator<(float4 const& u, float4 const& v)
{
    return mask4(
            u.value[0] < v.value[0],
            u.value[1] < v.value[1],
            u.value[2] < v.value[2],
            u.value[3] < v.value[3]
            );
}

VSNRAY_FORCE_INLINE mask4 operator>(float4 const& u, float4 const& v)
{
    return mask4(
            u.value[0] > v.value[0],
            u.value[1] > v.value[1],
            u.value[2] > v.value[2],
            u.value[3] > v.value[3]
            );
}

VSNRAY_FORCE_INLINE mask4 operator<=(float4 const& u, float4 const& v)
{
    return mask4(
            u.value[0] <= v.value[0],
            u.value[1] <= v.value[1],
            u.value[2] <= v.value[2],
            u.value[3] <= v.value[3]
            );
}

VSNRAY_FORCE_INLINE mask4 operator>=(float4 const& u, float4 const& v)
{
    return mask4(
            u.value[0] >= v.value[0],
            u.value[1] >= v.value[1],
            u.value[2] >= v.value[2],
            u.value[3] >= v.value[3]
            );
}

VSNRAY_FORCE_INLINE mask4 operator==(float4 const& u, float4 const& v)
{
    return mask4(
            u.value[0] == v.value[0],
            u.value[1] == v.value[1],
            u.value[2] == v.value[2],
            u.value[3] == v.value[3]
            );
}

VSNRAY_FORCE_INLINE mask4 operator!=(float4 const& u, float4 const& v)
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

VSNRAY_FORCE_INLINE float4 dot(float4 const& u, float4 const& v)
{
    return float4(u.value[0] * v.value[0] + u.value[1] * v.value[1] + u.value[2] * v.value[2] + u.value[3] * v.value[3]);
}

VSNRAY_FORCE_INLINE float4 min(float4 const& u, float4 const& v)
{
    return float4(
            std::min(u.value[0], v.value[0]),
            std::min(u.value[1], v.value[1]),
            std::min(u.value[2], v.value[2]),
            std::min(u.value[3], v.value[3])
            );
}

VSNRAY_FORCE_INLINE float4 max(float4 const& u, float4 const& v)
{
    return float4(
            std::max(u.value[0], v.value[0]),
            std::max(u.value[1], v.value[1]),
            std::max(u.value[2], v.value[2]),
            std::max(u.value[3], v.value[3])
            );
}

VSNRAY_FORCE_INLINE float4 saturate(float4 const& u)
{
    return max(float4(0.0f), min(u, float4(0.0f)));
}

VSNRAY_FORCE_INLINE float4 abs(float4 const& u)
{
    return float4(
            std::abs(u.value[0]),
            std::abs(u.value[1]),
            std::abs(u.value[2]),
            std::abs(u.value[3])
            );
}

VSNRAY_FORCE_INLINE float4 round(float4 const& v)
{
    return float4(
            std::round(v.value[0]),
            std::round(v.value[1]),
            std::round(v.value[2]),
            std::round(v.value[3])
            );
}

VSNRAY_FORCE_INLINE float4 ceil(float4 const& v)
{
    return float4(
            std::ceil(v.value[0]),
            std::ceil(v.value[1]),
            std::ceil(v.value[2]),
            std::ceil(v.value[3])
            );
}

VSNRAY_FORCE_INLINE float4 floor(float4 const& v)
{
    return float4(
            std::floor(v.value[0]),
            std::floor(v.value[1]),
            std::floor(v.value[2]),
            std::floor(v.value[3])
            );
}

VSNRAY_FORCE_INLINE float4 sqrt(float4 const& v)
{
    return float4(
            std::sqrt(v.value[0]),
            std::sqrt(v.value[1]),
            std::sqrt(v.value[2]),
            std::sqrt(v.value[3])
            );
}

VSNRAY_FORCE_INLINE mask4 isinf(float4 const& v)
{
    return mask4(
            std::isinf(v.value[0]),
            std::isinf(v.value[1]),
            std::isinf(v.value[2]),
            std::isinf(v.value[3])
            );
}

VSNRAY_FORCE_INLINE mask4 isnan(float4 const& v)
{
    return mask4(
            std::isnan(v.value[0]),
            std::isnan(v.value[1]),
            std::isnan(v.value[2]),
            std::isnan(v.value[3])
            );
}

VSNRAY_FORCE_INLINE mask4 isfinite(float4 const& v)
{
    return mask4(
            std::isfinite(v.value[0]),
            std::isfinite(v.value[1]),
            std::isfinite(v.value[2]),
            std::isfinite(v.value[3])
            );
}


//-------------------------------------------------------------------------------------------------
//
//

VSNRAY_FORCE_INLINE float4 rcp(float4 const& v)
{
    return float4(1.0f) / v;
}

VSNRAY_FORCE_INLINE float4 rsqrt(float4 const& v)
{
    return float4(1.0f) / sqrt(v);
}

} // simd
} // MATH_NAMESPACE
