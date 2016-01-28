// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <type_traits>

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

template <size_t N>
inline vector<4, typename float_from_simd_width<N>::type> pack(
        std::array<vector<4, float>, N> const& vecs
        )
{
    using T = typename float_from_simd_width<N>::type;
    using float_array = typename simd::aligned_array<T>::type;

    float_array x;
    float_array y;
    float_array z;
    float_array w;

    for (size_t i = 0; i < N; ++i)
    {
        x[i] = vecs[i].x;
        y[i] = vecs[i].y;
        z[i] = vecs[i].z;
        w[i] = vecs[i].w;
    }
    return vector<4, float4>(x, y, z, w);
}

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
