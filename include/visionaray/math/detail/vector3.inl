// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

#include <visionaray/array.h>

#include "../simd/type_traits.h"
#include "math.h"

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// vector3 members
//

template <typename T>
MATH_FUNC
inline vector<3, T>::vector(T const& x, T const& y, T const& z)
    : x(x)
    , y(y)
    , z(z)
{
}

template <typename T>
MATH_FUNC
inline vector<3, T>::vector(T const& s)
    : x(s)
    , y(s)
    , z(s)
{
}

template <typename T>
MATH_FUNC
inline vector<3, T>::vector(T const data[3])
    : x(data[0])
    , y(data[1])
    , z(data[2])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<3, T>::vector(vector<2, U> const& rhs, U const& z)
    : x(rhs.x)
    , y(rhs.y)
    , z(z)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<3, T>::vector(vector<3, U> const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<3, T>::vector(vector<4, U> const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline vector<3, T>& vector<3, T>::operator=(vector<3, U> const& rhs)
{

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    return *this;

}

template <typename T>
MATH_FUNC
inline T* vector<3, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* vector<3, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline T& vector<3, T>::operator[](size_t i)
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline T const& vector<3, T>::operator[](size_t i) const
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline vector<2, T>& vector<3, T>::xy()
{
    return *reinterpret_cast<vector<2, T>*>( data() );
}

template <typename T>
MATH_FUNC
inline vector<2, T> const& vector<3, T>::xy() const
{
    return *reinterpret_cast<vector<2, T> const*>( data() );
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline vector<3, T> operator+(vector<3, T> const& v)
{
    return vector<3, T>(+v.x, +v.y, +v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator-(vector<3, T> const& v)
{
    return vector<3, T>(-v.x, -v.y, -v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator+(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator-(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator*(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>(u.x * v.x, u.y * v.y, u.z * v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator/(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>(u.x / v.x, u.y / v.y, u.z / v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator+(vector<3, T> const& v, T const& s)
{
    return vector<3, T>(v.x + s, v.y + s, v.z + s);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator-(vector<3, T> const& v, T const& s)
{
    return vector<3, T>(v.x - s, v.y - s, v.z - s);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator*(vector<3, T> const& v, T const& s)
{
    return vector<3, T>(v.x * s, v.y * s, v.z * s);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator/(vector<3, T> const& v, T const& s)
{
    return vector<3, T>(v.x / s, v.y / s, v.z / s);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator+(T const& s, vector<3, T> const& v)
{
    return vector<3, T>(s + v.x, s + v.y, s + v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator-(T const& s, vector<3, T> const& v)
{
    return vector<3, T>(s - v.x, s - v.y, s - v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator*(T const& s, vector<3, T> const& v)
{
    return vector<3, T>(s * v.x, s * v.y, s * v.z);
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator/(T const& s, vector<3, T> const& v)
{
    return vector<3, T>(s / v.x, s / v.y, s / v.z);
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
inline auto operator==(vector<3, T> const& u, vector<3, T> const& v)
    -> decltype(u.x == v.x)
{
    return u.x == v.x && u.y == v.y && u.z == v.z;
}

template <typename T>
MATH_FUNC
inline auto operator<(vector<3, T> const& u, vector<3, T> const& v)
    -> decltype(u.x < v.x)
{
    return u.x < v.x || ( (u.x == v.x && u.y < v.y) || (u.y == v.y && u.z < v.z) );
}

template <typename T>
MATH_FUNC
inline auto operator!=(vector<3, T> const& u, vector<3, T> const& v)
    -> decltype(u == v)
{
    return !(u == v);
}

template <typename T>
MATH_FUNC
inline auto operator<=(vector<3, T> const& u, vector<3, T> const& v)
    -> decltype(v < u)
{
    return !(v < u);
}

template <typename T>
MATH_FUNC
inline auto operator>(vector<3, T> const& u, vector<3, T> const& v)
    -> decltype(v < u)
{
    return v < u;
}

template <typename T>
MATH_FUNC
inline auto operator>=(vector<3, T> const& u, vector<3, T> const& v)
    -> decltype(u < v)
{
    return !(u < v);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline vector<3, T> cross(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>
    (
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

template <typename T>
MATH_FUNC
inline T dot(vector<3, T> const& u, vector<3, T> const& v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

template <typename T>
MATH_FUNC
inline T norm(vector<3, T>  const& v)
{
    return sqrt( dot(v, v) );
}

template <typename T>
MATH_FUNC
inline T norm2(vector<3, T> const& v)
{
    return dot(v, v);
}

template <typename T>
MATH_FUNC
inline T length(vector<3, T> const& v)
{
    return norm(v);
}

template <typename T>
MATH_FUNC
inline vector<3, T> normalize(vector<3, T> const& v)
{
    return v * rsqrt( dot(v, v) );
}


//-------------------------------------------------------------------------------------------------
// make_orthonormal_basis
//
// Make an orthonormal basis over 3D vectors u,v,w, where u and v are (bi)tangent
// and w is normal.
//
// Parameters:
//
// [out] U
//      Vector in the tangent plane.
//
// [out] V
//      Vector in the tangent plane.
//
// [in] W
//      Normal vector.
//

template <typename T>
MATH_FUNC
inline void make_orthonormal_basis(vector<3, T>& u, vector<3, T>& v, vector<3, T> const& w)
{
    v = select(
            abs(w.x) > abs(w.y),
            normalize( vector<3, T>(-w.z, T(0.0), w.x) ),
            normalize( vector<3, T>(T(0.0), w.z, -w.y) )
            );
    u = cross(v, w);
}


//--------------------------------------------------------------------------------------------------
// Misc.
//

template <typename M, typename T>
MATH_FUNC
inline vector<3, T> select(M const& m, vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>(
            select(m, u.x, v.x),
            select(m, u.y, v.y),
            select(m, u.z, v.z)
            );
}

template <typename M, typename T1, typename T2>
MATH_FUNC
auto select(M const& m, vector<3, T1> const& u, vector<3, T2> const& v)
    -> vector<3, decltype(select(m, u.x, v.x))>
{
    using T3 = decltype(select(m, u.x, v.x));

    return vector<3, T3>(
            select(m, u.x, v.x),
            select(m, u.y, v.y),
            select(m, u.z, v.z)
            );
}

template <typename T>
MATH_FUNC
inline vector<3, T> min(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>( min(u.x, v.x), min(u.y, v.y), min(u.z, v.z) );
}

template <typename T>
MATH_FUNC
inline vector<3, T> max(vector<3, T> const& u, vector<3, T> const& v)
{
    return vector<3, T>( max(u.x, v.x), max(u.y, v.y), max(u.z, v.z) );
}

template <typename T>
MATH_FUNC
inline T hadd(vector<3, T> const& u)
{
    return u.x + u.y + u.z;
}

template <typename T>
MATH_FUNC
inline T hmul(vector<3, T> const& u)
{
    return u.x * u.y * u.z;
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

template <typename T, size_t N>
MATH_FUNC
inline vector<3, float_from_simd_width_t<N>> pack(array<vector<3, T>, N> const& vecs)
{
    using U = float_from_simd_width_t<N>; // TODO: generalize, not just float!

    vector<3, U> result;

    T* x = reinterpret_cast<T*>(&result.x);
    T* y = reinterpret_cast<T*>(&result.y);
    T* z = reinterpret_cast<T*>(&result.z);

    for (size_t i = 0; i < N; ++i)
    {
        x[i] = vecs[i].x;
        y[i] = vecs[i].y;
        z[i] = vecs[i].z;
    }

    return result;
}

// unpack -------------------------------------------------

template <
    typename T,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type
    >
MATH_FUNC
inline array<vector<3, element_type_t<T>>, num_elements<T>::value> unpack(vector<3, T> const& v)
{
    using U = element_type_t<T>;

    U const* x = reinterpret_cast<U const*>(&v.x);
    U const* y = reinterpret_cast<U const*>(&v.y);
    U const* z = reinterpret_cast<U const*>(&v.z);

    array<vector<3, U>, num_elements<T>::value> result;

    for (int i = 0; i < num_elements<T>::value; ++i)
    {
        result[i].x = x[i];
        result[i].y = y[i];
        result[i].z = z[i];
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
