// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// vectorN members

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>::vector(T s)
{
    for (size_t d = 0; d < Dim; ++d)
    {
        data_[d] = s;
    }
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>::vector(T const data[Dim])
{
    std::copy( data, data + Dim, data_ );    
}

template <size_t Dim, typename T>
template <typename U>
MATH_FUNC
inline vector<Dim, T>::vector(vector<Dim, U> const& rhs)
{
    std::copy( rhs.data(), rhs.data() + Dim, data_ );
}

template <size_t Dim, typename T>
template <typename U>
MATH_FUNC
inline vector<Dim, T>& vector<Dim, T>::operator=(vector<Dim, U> const& rhs)
{
    std::copy( rhs.data(), rhs.data() + Dim, data_ );
    return *this;
}

template <size_t Dim, typename T>
MATH_FUNC
inline size_t vector<Dim, T>::size() const
{
    return Dim;
}

template <size_t Dim, typename T>
MATH_FUNC
inline T* vector<Dim, T>::data()
{
    return data_;
}

template <size_t Dim, typename T>
MATH_FUNC
inline T const* vector<Dim, T>::data() const
{
    return data_;
}

template <size_t Dim, typename T>
MATH_FUNC
inline T& vector<Dim, T>::operator[](size_t i)
{
    return data_[i];
}

template <size_t Dim, typename T>
MATH_FUNC
inline T const& vector<Dim, T>::operator[](size_t i) const
{
    return data_[i];
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator-(vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = -v[d];
    }

    return result;
}

// vector op vector

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator+(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] + v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator-(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] - v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator*(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] * v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator/(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] / v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator&(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] & v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator|(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] | v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator^(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] ^ v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator<<(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] << v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator>>(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] >> v[d];
    }

    return result;
}

// vector op scalar

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator+(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] + s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator-(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] - s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator*(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] * s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator/(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] / s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator&(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] & s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator|(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] | s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator^(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] ^ s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator<<(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] << s;
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator>>(vector<Dim, T> const& v, T s)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] >> s;
    }

    return result;
}

// scalar op vector

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator+(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s + v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator-(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s - v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator*(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s * v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator/(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s / v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator&(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s & v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator|(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s | v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator^(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s ^ v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator<<(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s << v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> operator>>(T s, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = s >> v[d];
    }

    return result;
}

// append operations

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator+=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u + v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator+=(vector<Dim, T>& v, T s)
{
    v = v + s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator-=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u - v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator-=(vector<Dim, T>& v, T s)
{
    v = v - s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator*=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u * v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator*=(vector<Dim, T>& v, T s)
{
    v = v * s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator/=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u / v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator/=(vector<Dim, T>& v, T s)
{
    v = v / s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator&=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u & v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator&=(vector<Dim, T>& v, T s)
{
    v = v & s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator|=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u | v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator|=(vector<Dim, T>& v, T s)
{
    v = v | s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator^=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u ^ v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator^=(vector<Dim, T>& v, T s)
{
    v = v ^ s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator<<=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u << v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator<<=(vector<Dim, T>& v, T s)
{
    v = v << s;
    return v;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator>>=(vector<Dim, T>& u, vector<Dim, T> const& v)
{
    u = u >> v;
    return u;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>& operator>>=(vector<Dim, T>& v, T s)
{
    v = v >> s;
    return v;
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> dot(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    T result(0.0);

    for (size_t d = 0; d < Dim; ++d)
    {
        result += u[d] * v[d];
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> reflect(vector<Dim, T> const& i, vector<Dim, T> const& n)
{
    return i - T(2.0) * dot(n, i) * n;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> refract(vector<Dim, T> const& i, vector<Dim, T> const& n, T eta)
{
    T k = T(1.0) - eta * eta * (T(1.0) - dot(n, i) * dot(n, i));
    if (k < T(0.0))
    {
        return vector<Dim, T>(0.0);
    }
    else
    {
        return eta * i - (eta * dot(n, i) + sqrt(k)) * n;
    }
}


//-------------------------------------------------------------------------------------------------
// Misc.
//

template <size_t Dim, typename M, typename T>
MATH_FUNC
inline vector<Dim, T> select(M const& m, vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = select(m, u[d], v[d]);
    }

    return result;
}

// Returns the index of the smallest element of the vector
template <size_t Dim, typename T>
size_t min_index(vector<Dim, T> const& u)
{
    size_t i = u.y < u.x ? 1 : 0;

    for (size_t n = 2; n < Dim; ++n)
    {
        i = u[n] < u[i] ? n : i;
    }

    return i;
}

// Returns the index of the largest element of the vector
template <size_t Dim, typename T>
size_t max_index(vector<Dim, T> const& u)
{
    size_t i = u.y < u.x ? 0 : 1;

    for (size_t n = 2; n < Dim; ++n)
    {
        i = u[n] < u[i] ? i : n;
    }

    return i;
}

template <size_t Dim, typename T>
MATH_FUNC
T hadd(vector<Dim, T> const& u)
{
    T result(0.0);

    for (size_t d = 0; d < Dim; ++d)
    {
        result += u[d];
    }

    return result;
}

} // MATH_NAMESPACE
