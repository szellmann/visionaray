// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <type_traits>

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
#include "../simd/avx.h"
#endif
#include "../simd/sse.h"
#include "../simd/type_traits.h"

namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// vector4 members
//

template <typename T>
MATH_FUNC
inline vector<4, T>::vector(T x, T y, T z, T w)
    : x(x)
    , y(y)
    , z(z)
    , w(w)
{
}

template <typename T>
MATH_FUNC
inline vector<4, T>::vector(T s)
    : x(s)
    , y(s)
    , z(s)
    , w(s)
{
}

template <typename T>
MATH_FUNC
inline vector<4, T>::vector(T const data[4])
    : x(data[0])
    , y(data[1])
    , z(data[2])
    , w(data[3])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<4, T>::vector(vector<2, U> const& rhs, U z, U w)
    : x(rhs.x)
    , y(rhs.y)
    , z(z)
    , w(w)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<4, T>::vector(vector<3, U> const& rhs, U w)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
    , w(w)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<4, T>::vector(vector<4, U> const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
    , w(rhs.w)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<4, T>& vector<4, T>::operator=(vector<4, U> const& rhs)
{

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;
    return *this;

}

template <typename T>
MATH_FUNC
inline T* vector<4, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* vector<4, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline T& vector<4, T>::operator[](size_t i)
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline T const& vector<4, T>::operator[](size_t i) const
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline vector<3, T>& vector<4, T>::xyz()
{
    return *reinterpret_cast<vector<3, T>*>( data() );
}

template <typename T>
MATH_FUNC
inline vector<3, T> const& vector<4, T>::xyz() const
{
    return *reinterpret_cast<vector<3, T> const*>( data() );
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline vector<4, T> operator-(vector<4, T> const& v)
{
    return vector<4, T>(-v.x, -v.y, -v.z, -v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator+(vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator-(vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator*(vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator/(vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator+(vector<4, T> const& v, T const& s)
{
    return vector<4, T>(v.x + s, v.y + s, v.z + s, v.w + s);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator-(vector<4, T> const& v, T const& s)
{
    return vector<4, T>(v.x - s, v.y - s, v.z - s, v.w - s);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator*(vector<4, T> const& v, T const& s)
{
    return vector<4, T>(v.x * s, v.y * s, v.z * s, v.w * s);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator/(vector<4, T> const& v, T const& s)
{
    return vector<4, T>(v.x / s, v.y / s, v.z / s, v.w / s);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator+(T const& s, vector<4, T> const& v)
{
    return vector<4, T>(s + v.x, s + v.y, s + v.z, s + v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator-(T const& s, vector<4, T> const& v)
{
    return vector<4, T>(s - v.x, s - v.y, s - v.z, s - v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator*(T const& s, vector<4, T> const& v)
{
    return vector<4, T>(s * v.x, s * v.y, s * v.z, s * v.w);
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator/(T const& s, vector<4, T> const& v)
{
    return vector<4, T>(s / v.x, s / v.y, s / v.z, s / v.w);
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
bool operator==(vector<4, T> const& u, vector<4, T> const& v)
{
    return u.x == v.x && u.y == v.y && u.z == v.z && u.w == v.w;
}

template <typename T>
MATH_FUNC
bool operator<(vector<4, T> const& u, vector<4, T> const& v)
{
    return u.x < v.x || ( (u.x == v.x && u.y < v.y) || ( (u.y == v.y && u.z < v.z) || (u.z == v.z && u.w < v.w) ) );
}

template <typename T>
MATH_FUNC
bool operator!=(vector<4, T> const& u, vector<4, T> const& v)
{
    return !(u == v);
}

template <typename T>
MATH_FUNC
bool operator<=(vector<4, T> const& u, vector<4, T> const& v)
{
    return !(v < u);
}

template <typename T>
MATH_FUNC
bool operator>(vector<4, T> const& u, vector<4, T> const& v)
{
    return v < u;
}

template <typename T>
MATH_FUNC
bool operator>=(vector<4, T> const& u, vector<4, T> const& v)
{
    return !(u < v);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline T dot(vector<4, T> const& u, vector<4, T> const& v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

template <typename T>
MATH_FUNC
inline T norm(vector<4, T> const& v)
{
    return sqrt( dot(v, v) );
}

template <typename T>
MATH_FUNC
inline T norm2(vector<4, T> const& v)
{
    return dot(v, v);
}

template <typename T>
MATH_FUNC
inline T length(vector<4, T> const& v)
{
    return norm(v);
}

template <typename T>
MATH_FUNC
inline vector<4, T> normalize(vector<4, T> const& v)
{
    return v * rsqrt( dot(v, v) );
}


//-------------------------------------------------------------------------------------------------
// Misc.
//

template <typename M, typename T>
MATH_FUNC
vector<4, T> select(M const& m, vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>(
            select(m, u.x, v.x),
            select(m, u.y, v.y),
            select(m, u.z, v.z),
            select(m, u.w, v.w)
            );
}

template <typename M, typename T1, typename T2>
MATH_FUNC
auto select(M const& m, vector<4, T1> const& u, vector<4, T2> const& v)
    -> vector<4, decltype(select(m, u.x, v.x))>
{
    using T3 = decltype(select(m, u.x, v.x));

    return vector<4, T3>(
            select(m, u.x, v.x),
            select(m, u.y, v.y),
            select(m, u.z, v.z),
            select(m, u.w, v.w)
            );
}

template <typename T>
MATH_FUNC
inline vector<4, T> min(vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>( min(u.x, v.x), min(u.y, v.y), min(u.z, v.z), min(u.w, v.w) );
}

template <typename T>
MATH_FUNC
inline vector<4, T> max(vector<4, T> const& u, vector<4, T> const& v)
{
    return vector<4, T>( max(u.x, v.x), max(u.y, v.y), max(u.z, v.z), max(u.w, v.w) );
}

template <typename T>
MATH_FUNC
inline T hadd(vector<4, T> const& u)
{
    return u.x + u.y + u.z + u.w;
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

inline vector<4, float4> pack(std::array<vector<4, float>, 4> const& vecs)
{
    return vector<4, float4>(
            float4(vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x),
            float4(vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y),
            float4(vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z),
            float4(vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w)
            );
}

inline vector<4, float4> pack(
        vector<4, float> const& v1,
        vector<4, float> const& v2,
        vector<4, float> const& v3,
        vector<4, float> const& v4
        )
{
    return vector<4, float4>(
            float4(v1.x, v2.x, v3.x, v4.x),
            float4(v1.y, v2.y, v3.y, v4.y),
            float4(v1.z, v2.z, v3.z, v4.z),
            float4(v1.w, v2.w, v3.w, v4.w)
            );
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

inline vector<4, float8> pack(std::array<vector<4, float>, 8> const& vecs)
{
    return vector<4, float8>(
            float8(vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x, vecs[4].x, vecs[5].x, vecs[6].x, vecs[7].x),
            float8(vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y, vecs[4].y, vecs[5].y, vecs[6].y, vecs[7].y),
            float8(vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z, vecs[4].z, vecs[5].z, vecs[6].z, vecs[7].z),
            float8(vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w, vecs[4].w, vecs[5].w, vecs[6].w, vecs[7].w)
            );
}

inline vector<4, float8> pack(
        vector<4, float> const& v1,
        vector<4, float> const& v2,
        vector<4, float> const& v3,
        vector<4, float> const& v4,
        vector<4, float> const& v5,
        vector<4, float> const& v6,
        vector<4, float> const& v7,
        vector<4, float> const& v8
        )
{
    return vector<4, float8>(
            float8(v1.x, v2.x, v3.x, v4.x, v5.x, v6.x, v7.x, v8.x),
            float8(v1.y, v2.y, v3.y, v4.y, v5.y, v6.y, v7.y, v8.y),
            float8(v1.z, v2.z, v3.z, v4.z, v5.z, v6.z, v7.z, v8.z),
            float8(v1.w, v2.w, v3.w, v4.w, v5.w, v6.w, v7.w, v8.w)
            );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// unpack -------------------------------------------------

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline std::array<vector<4, float>, num_elements<FloatT>::value> unpack(
        vector<4, FloatT> const& v
        )
{
    using float_array = typename aligned_array<FloatT>::type;

    float_array x;
    float_array y;
    float_array z;
    float_array w;

    store(x, v.x);
    store(y, v.y);
    store(z, v.z);
    store(w, v.w);

    std::array<vector<4, float>, num_elements<FloatT>::value> result;

    for (int i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].x = x[i];
        result[i].y = y[i];
        result[i].z = z[i];
        result[i].w = w[i];
    }

    return result;
}


// Transpose to get from SoA to AoS (and vice versa)
// Similar to mat4 transpose
inline vector<4, float4> transpose(vector<4, float4> const& v)
{

    float4 tmp0 = _mm_unpacklo_ps(v.x, v.y);
    float4 tmp1 = _mm_unpacklo_ps(v.z, v.w);
    float4 tmp2 = _mm_unpackhi_ps(v.x, v.y);
    float4 tmp3 = _mm_unpackhi_ps(v.z, v.w);

    return vector<4, float4>
    (
        _mm_movelh_ps(tmp0, tmp1),
        _mm_movehl_ps(tmp1, tmp0),
        _mm_movelh_ps(tmp2, tmp3),
        _mm_movehl_ps(tmp3, tmp2)
    );

}

// TODO: transpose for AVX?

} // simd

} // MATH_NAMESPACE
