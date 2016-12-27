// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <array>
#include <type_traits>

#include "../simd/type_traits.h"
#include "math.h"

namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// vectorN members

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T>::vector(T const& s)
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
template <size_t Dim1, size_t Dim2, typename U>
MATH_FUNC
inline vector<Dim, T>::vector(vector<Dim1, U> const& first, vector<Dim2, U> const& second)
{
    static_assert(Dim1 + Dim2 == Dim, "Incompatible vector dimensions");
    std::copy( first.data(),  first.data()  + Dim1, data_ );
    std::copy( second.data(), second.data() + Dim2, data_ + Dim1 );
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
inline vector<Dim, T> operator+(vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = +v[d];
    }

    return result;
}

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
inline vector<Dim, T> operator+(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator-(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator*(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator/(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator&(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator|(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator^(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator<<(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator>>(vector<Dim, T> const& v, T const& s)
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
inline vector<Dim, T> operator+(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator-(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator*(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator/(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator&(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator|(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator^(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator<<(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T> operator>>(T const& s, vector<Dim, T> const& v)
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
inline vector<Dim, T>& operator+=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator-=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator*=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator/=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator&=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator|=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator^=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator<<=(vector<Dim, T>& v, T const& s)
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
inline vector<Dim, T>& operator>>=(vector<Dim, T>& v, T const& s)
{
    v = v >> s;
    return v;
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <size_t Dim, typename T>
MATH_FUNC
inline T dot(vector<Dim, T> const& u, vector<Dim, T> const& v)
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
inline vector<Dim, T> refract(vector<Dim, T> const& i, vector<Dim, T> const& n, T const& eta)
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
inline vector<Dim, T> min(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = min(u[d], v[d]);
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> max(vector<Dim, T> const& u, vector<Dim, T> const& v)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = max(u[d], v[d]);
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline vector<Dim, T> rotate(vector<Dim, T> v, int count = 1)
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
inline size_t min_index(vector<Dim, T> const& u)
{
    size_t i = u[1] < u[0] ? 1 : 0;

    for (size_t n = 2; n < Dim; ++n)
    {
        i = u[n] < u[i] ? n : i;
    }

    return i;
}

// Returns the index of the largest element of the vector
template <size_t Dim, typename T>
MATH_FUNC
inline size_t max_index(vector<Dim, T> const& u)
{
    size_t i = u[1] < u[0] ? 0 : 1;

    for (size_t n = 2; n < Dim; ++n)
    {
        i = u[n] < u[i] ? i : n;
    }

    return i;
}

// Returns the indices of the smallest and largest elements of the vector
template <size_t Dim, typename T>
MATH_FUNC
inline vector<2, size_t> minmax_index(vector<Dim, T> const& u)
{
    vector<2, size_t> i(
            u[1] < u[0] ? 1 : 0,
            u[1] < u[0] ? 0 : 1
            );

    for (size_t n = 2; n < Dim; ++n)
    {
        i.x = u[n] < u[i.x] ? n : i.x;
        i.y = u[n] < u[i.y] ? i.y : n;
    }

    return i;
}

// Returns the smallest element
template <size_t Dim, typename T>
MATH_FUNC
inline T min_element(vector<Dim, T> const& u)
{
    T result = u[0];

    for (size_t n = 1; n < Dim; ++n)
    {
        result = select(
                u[n] < result,
                u[n],
                result
                );
    }

    return result;
}

// Returns the largest element
template <size_t Dim, typename T>
MATH_FUNC
inline T max_element(vector<Dim, T> const& u)
{
    T result = u[0];

    for (size_t n = 1; n < Dim; ++n)
    {
        result = select(
                u[n] > result,
                u[n],
                result
                );
    }

    return result;
}

// Returns the smallest and the largest element
template <size_t Dim, typename T>
MATH_FUNC
inline vector<2, T> minmax_element(vector<Dim, T> const& u)
{
    vector<2, T> result(u[0]);

    for (size_t n = 1; n < Dim; ++n)
    {
        result.x = select(
                u[n] < result.x,
                u[n],
                result.x
                );

        result.y = select(
                u[n] > result.y,
                u[n],
                result.y
                );
    }

    return result;
}

template <size_t Dim, typename T>
MATH_FUNC
inline T hadd(vector<Dim, T> const& u)
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
inline T hmul(vector<Dim, T> const& u)
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
inline auto convert_to_float(vector<Dim, T> const& v)
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
inline auto convert_to_int(vector<Dim, T> const& v)
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

// pack ---------------------------------------------------

template <size_t Dim, typename T, size_t N> // TODO: check that T is convertible to float
inline auto pack(std::array<vector<Dim, T>, N> const& vecs)
    -> vector<Dim, float_from_simd_width_t<N>>
{
    using U = float_from_simd_width_t<N>;
    using float_array = aligned_array_t<U>;

    vector<Dim, U> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        float_array v;
        for (size_t i = 0; i < N; ++i)
        {
            v[i] = vecs[i][d];
        }
        result[d] = U(v);
    }

    return result;
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE2

// pack four vectors

template <size_t Dim, typename T>
inline auto pack(
        vector<Dim, T> const& v1,
        vector<Dim, T> const& v2,
        vector<Dim, T> const& v3,
        vector<Dim, T> const& v4
        )
    -> vector<Dim, float_from_simd_width_t<4>>
{
    return pack( std::array<vector<Dim, T>, 4>{{
            v1, v2, v3, v4
            }} );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_SSE2

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// pack eight vectors

template <size_t Dim, typename T>
inline auto pack(
        vector<Dim, T> const& v1,
        vector<Dim, T> const& v2,
        vector<Dim, T> const& v3,
        vector<Dim, T> const& v4,
        vector<Dim, T> const& v5,
        vector<Dim, T> const& v6,
        vector<Dim, T> const& v7,
        vector<Dim, T> const& v8
        )
    -> vector<Dim, float_from_simd_width_t<8>>
{
    return pack( std::array<vector<Dim, T>, 8>{{
            v1, v2, v3, v4, v5, v6, v7, v8
            }} );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// unpack -------------------------------------------------

template <
    size_t Dim,
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline auto unpack(vector<Dim, FloatT> const& v)
    -> std::array<vector<Dim, float>, num_elements<FloatT>::value>
{
    using float_array = aligned_array_t<FloatT>;

    std::array<vector<Dim, float>, num_elements<FloatT>::value> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        float_array data;
        store(data, v[d]);

        for (int i = 0; i < num_elements<FloatT>::value; ++i)
        {
            result[i][d] = data[i];
        }
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
