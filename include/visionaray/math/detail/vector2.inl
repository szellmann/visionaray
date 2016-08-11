// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "math.h"

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// vector2 members
//

template <typename T>
MATH_FUNC
inline vector<2, T>::vector(T const& x, T const& y)
    : x(x)
    , y(y)
{
}

template <typename T>
MATH_FUNC
inline vector<2, T>::vector(T const& s)
    : x(s)
    , y(s)
{
}

template <typename T>
MATH_FUNC
inline vector<2, T>::vector(T const data[2])
    : x(data[0])
    , y(data[1])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<2, T>::vector(vector<2, U> const& rhs)
    : x(rhs.x)
    , y(rhs.y)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<2, T>::vector(vector<3, U> const& rhs)
    : x(rhs.x)
    , y(rhs.y)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<2, T>::vector(vector<4, U> const& rhs)
    : x(rhs.x)
    , y(rhs.y)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<2, T>& vector<2, T>::operator=(vector<2, U> const& rhs)
{

    x = rhs.x;
    y = rhs.y;
    return *this;

}

template <typename T>
MATH_FUNC
inline T* vector<2, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* vector<2, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline T& vector<2, T>::operator[](size_t i)
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline T const& vector<2, T>::operator[](size_t i) const
{
    return data()[i];
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline vector<2, T> operator+(vector<2, T> const& v)
{
    return vector<2, T>(+v.x, +v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator-(vector<2, T> const& v)
{
    return vector<2, T>(-v.x, -v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator+(vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>(u.x + v.x, u.y + v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator-(vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>(u.x - v.x, u.y - v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator*(vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>(u.x * v.x, u.y * v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator/(vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>(u.x / v.x, u.y / v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator+(vector<2, T> const& v, T const& s)
{
    return vector<2, T>(v.x + s, v.y + s);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator-(vector<2, T> const& v, T const& s)
{
    return vector<2, T>(v.x - s, v.y - s);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator*(vector<2, T> const& v, T const& s)
{
    return vector<2, T>(v.x * s, v.y * s);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator/(vector<2, T> const& v, T const& s)
{
    return vector<2, T>(v.x / s, v.y / s);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator+(T const& s, vector<2, T> const& v)
{
    return vector<2, T>(s + v.x, s + v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator-(T const& s, vector<2, T> const& v)
{
    return vector<2, T>(s - v.x, s - v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator*(T const& s, vector<2, T> const& v)
{
    return vector<2, T>(s * v.x, s * v.y);
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator/(T const& s, vector<2, T> const& v)
{
    return vector<2, T>(s / v.x, s / v.y);
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
inline auto operator==(vector<2, T> const& u, vector<2, T> const& v)
    -> decltype(u.x == v.x)
{
    return u.x == v.x && u.y == v.y;
}

template <typename T>
MATH_FUNC
inline auto operator<(vector<2, T> const& u, vector<2, T> const& v)
    -> decltype(u.x < v.x)
{
    return u.x < v.x || (u.x == v.x && u.y < v.y);
}

template <typename T>
MATH_FUNC
inline auto operator!=(vector<2, T> const& u, vector<2, T> const& v)
    -> decltype(u == v)
{
    return !(u == v);
}

template <typename T>
MATH_FUNC
inline auto operator<=(vector<2, T> const& u, vector<2, T> const& v)
    -> decltype(v < u)
{
    return !(v < u);
}

template <typename T>
MATH_FUNC
inline auto operator>(vector<2, T> const& u, vector<2, T> const& v)
    -> decltype(v < u)
{
    return v < u;
}

template <typename T>
MATH_FUNC
inline auto operator>=(vector<2, T> const& u, vector<2, T> const& v)
    -> decltype(u < v)
{
    return !(u < v);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline T dot(vector<2, T> const& u, vector<2, T> const& v)
{
    return u.x * v.x + u.y * v.y;
}

template <typename T>
MATH_FUNC
inline T norm(vector<2, T>  const& v)
{
    return sqrt( dot(v, v) );
}

template <typename T>
MATH_FUNC
inline T norm2(vector<2, T> const& v)
{
    return dot(v, v);
}

template <typename T>
MATH_FUNC
inline T length(vector<2, T> const& v)
{
    return norm(v);
}

template <typename T>
MATH_FUNC
inline vector<2, T> normalize(vector<2, T> const& v)
{
    return v * rsqrt( dot(v, v) );
}


//-------------------------------------------------------------------------------------------------
// Misc.
//

template <typename M, typename T>
MATH_FUNC
inline vector<2, T> select(M const& m, vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>(
            select(m, u.x, v.x),
            select(m, u.y, v.y)
            );
}

template <typename M, typename T1, typename T2>
MATH_FUNC
auto select(M const& m, vector<2, T1> const& u, vector<2, T2> const& v)
    -> vector<2, decltype(select(m, u.x, v.x))>
{
    using T3 = decltype(select(m, u.x, v.x));

    return vector<2, T3>(
            select(m, u.x, v.x),
            select(m, u.y, v.y)
            );
}

template <typename T>
MATH_FUNC
inline vector<2, T> min(vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>( min(u.x, v.x), min(u.y, v.y) );
}

template <typename T>
MATH_FUNC
inline vector<2, T> max(vector<2, T> const& u, vector<2, T> const& v)
{
    return vector<2, T>( max(u.x, v.x), max(u.y, v.y) );
}

template <typename T>
MATH_FUNC
inline vector<2, T> hadd(vector<2, T> const& u)
{
    return u.x + u.y;
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

template <typename T, size_t N> // TODO: check that T is convertible to float
inline vector<2, typename float_from_simd_width<N>::type> pack(
        std::array<vector<2, T>, N> const& vecs
        )
{
    using U = typename float_from_simd_width<N>::type;
    using float_array = typename aligned_array<U>::type;

    float_array x;
    float_array y;

    for (size_t i = 0; i < N; ++i)
    {
        x[i] = vecs[i].x;
        y[i] = vecs[i].y;
    }

    return vector<2, U>(x, y);
}

// unpack -------------------------------------------------

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline auto unpack(vector<2, FloatT> const& v)
    -> std::array<vector<2, float>, num_elements<FloatT>::value>
{
    using float_array = typename aligned_array<FloatT>::type;

    float_array x;
    float_array y;

    store(x, v.x);
    store(y, v.y);

    std::array<vector<2, float>, num_elements<FloatT>::value> result;

    for (int i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].x = x[i];
        result[i].y = y[i];
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
