// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// vector2 members
//

template <typename T>
MATH_FUNC
inline vector<2, T>::vector(T x, T y)
    : x(x)
    , y(y)
{
}

template <typename T>
MATH_FUNC
inline vector<2, T>::vector(T s)
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


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
inline bool operator==(vector<2, T> const& u, vector<2, T> const& v)
{
    return u.x == v.x && u.y == v.y;
}

template <typename T>
MATH_FUNC
inline bool operator<(vector<2, T> const& u, vector<2, T> const& v)
{
    return u.x < v.x || (u.x == v.x && u.y < v.y);
}

template <typename T>
MATH_FUNC
inline bool operator!=(vector<2, T> const& u, vector<2, T> const& v)
{
    return !(u == v);
}

template <typename T>
MATH_FUNC
inline bool operator<=(vector<2, T> const& u, vector<2, T> const& v)
{
    return !(v < u);
}

template <typename T>
MATH_FUNC
inline bool operator>(vector<2, T> const& u, vector<2, T> const& v)
{
    return v < u;
}

template <typename T>
MATH_FUNC
inline bool operator>=(vector<2, T> const& u, vector<2, T> const& v)
{
    return !(u < v);
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
    return vector<2, T>
    (
        select(m, u.x, v.x),
        select(m, u.y, v.y)
    );
}

template <typename T>
MATH_FUNC
inline vector<2, T> hadd(vector<2, T> const& u)
{
    return u.x + u.y;
}

} // MATH_NAMESPACE
