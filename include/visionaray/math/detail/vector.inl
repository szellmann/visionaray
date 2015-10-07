// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <array>

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
#include "../simd/avx.h"
#endif
#include "../simd/sse.h"

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
inline vector<Dim, T>::vector(T const* data/*[Dim]*/)
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
inline vector<Dim, T> faceforward(
        vector<Dim, T> const& n,
        vector<Dim, T> const& i,
        vector<Dim, T> const& nref
        )
{
    return select( dot(nref, i) < T(0.0), -n, n );
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> reflect(vector<Dim, T> const& i, vector<Dim, T> const& n)
{
    return T(2.0) * dot(n, i) * n - i;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> refract(vector<Dim, T> const& i, vector<Dim, T> const& n, T eta)
{
    T k = T(1.0) - eta * eta * (T(1.0) - dot(n, i) * dot(n, i));

    return select(
            k < T(0.0),
            vector<Dim, T>(0.0),
            (sqrt(k) - eta * dot(n, i)) * n - eta * i
            );
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> floor(vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = floor(v[d]);
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> ceil(vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = ceil(v[d]);
    }

    return result;
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

template <size_t Dim, typename T>
MATH_FUNC
vector<Dim, T> rotate(vector<Dim, T> v, int count = 1)
{
    auto tmp = v;

    for (size_t d = 0; d < Dim; ++d)
    {
        auto d2 = (d + count) % Dim;
        v[d2] = tmp[d];
    }

    return v;
}

// Returns the index of the smallest element of the vector
template <size_t Dim, typename T>
MATH_FUNC
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
MATH_FUNC
size_t max_index(vector<Dim, T> const& u)
{
    size_t i = u.y < u.x ? 0 : 1;

    for (size_t n = 2; n < Dim; ++n)
    {
        i = u[n] < u[i] ? i : n;
    }

    return i;
}

// Returns the smallest element
template <size_t Dim, typename T>
MATH_FUNC
T min_element(vector<Dim, T> const& u)
{
    T result = u.x;

    for (size_t n = 1; n < Dim; ++n)
    {
        result = u[n] < result ? u[n] : result;
    }

    return result;
}

// Returns the largest element
template <size_t Dim, typename T>
MATH_FUNC
T max_element(vector<Dim, T> const& u)
{
    T result = u.x;

    for (size_t n = 1; n < Dim; ++n)
    {
        result = u[n] > result ? u[n] : result;
    }

    return result;
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

template <size_t Dim, typename T>
MATH_FUNC
T hmul(vector<Dim, T> const& u)
{
    T result(1.0);

    for (size_t d = 0; d < Dim; ++d)
    {
        result *= u[d];
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Convert from float to int
//

template <size_t Dim, typename T>
MATH_FUNC
auto convert_to_float(vector<Dim, T> const& v)
    -> vector<Dim, decltype(convert_to_float(v.x))>
{
    vector<Dim, decltype(convert_to_float(v.x))> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = convert_to_float(v[d]);
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
auto convert_to_int(vector<Dim, T> const& v)
    -> vector<Dim, decltype(convert_to_int(v.x))>
{
    vector<Dim, decltype(convert_to_int(v.x))> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = convert_to_int(v[d]);
    }

    return result;
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// SSE

template <size_t Dim>
inline vector<Dim, float4> pack(
        vector<Dim, float> const& v1,
        vector<Dim, float> const& v2,
        vector<Dim, float> const& v3,
        vector<Dim, float> const& v4
        )
{
    vector<Dim, float4> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = float4( v1[d], v2[d], v3[d], v4[d] );
    }

    return result;
}

template <size_t Dim>
inline std::array<vector<Dim, float>, 4> unpack(vector<Dim, float4> const& v)
{
    std::array<vector<Dim, float>, 4> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        VSNRAY_ALIGN(16) float data[4];
        store(data, v[d]);

        result[0][d] = data[0];
        result[1][d] = data[1];
        result[2][d] = data[2];
        result[3][d] = data[3];
    }

    return result;
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX

template <size_t Dim>
inline vector<Dim, float8> pack(
        vector<Dim, float> const& v1,
        vector<Dim, float> const& v2,
        vector<Dim, float> const& v3,
        vector<Dim, float> const& v4,
        vector<Dim, float> const& v5,
        vector<Dim, float> const& v6,
        vector<Dim, float> const& v7,
        vector<Dim, float> const& v8
        )
{
    vector<Dim, float8> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = float8(
                v1[d],
                v2[d],
                v3[d],
                v4[d],
                v5[d],
                v6[d],
                v7[d],
                v8[d]
                );
    }

    return result;
}

template <size_t Dim>
inline std::array<vector<Dim, float>, 8> unpack(vector<Dim, float8> const& v)
{
    std::array<vector<Dim, float>, 8> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        VSNRAY_ALIGN(32) float data[8];
        store(data, v[d]);

        result[0][d] = data[0];
        result[1][d] = data[1];
        result[2][d] = data[2];
        result[3][d] = data[3];
        result[4][d] = data[4];
        result[5][d] = data[5];
        result[6][d] = data[6];
        result[7][d] = data[7];
    }

    return result;
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd

} // MATH_NAMESPACE
