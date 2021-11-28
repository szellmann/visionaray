// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../config.h"

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// matrix4x3 members
//

template <typename T>
MATH_FUNC
inline matrix<4, 3, T>::matrix(
        vector<3, T> const& c0,
        vector<3, T> const& c1,
        vector<3, T> const& c2,
        vector<3, T> const& c3
        )
    : col0(c0)
    , col1(c1)
    , col2(c2)
    , col3(c3)
{
}

template <typename T>
MATH_FUNC
inline matrix<4, 3, T>::matrix(
        T const& m00, T const& m10, T const& m20,
        T const& m01, T const& m11, T const& m21,
        T const& m02, T const& m12, T const& m22,
        T const& m03, T const& m13, T const& m23
        )
    : col0(m00, m10, m20)
    , col1(m01, m11, m21)
    , col2(m02, m12, m22)
    , col3(m03, m13, m23)
{
}

template <typename T>
MATH_FUNC
inline matrix<4, 3, T>::matrix(matrix<3, 3, T> const& m, vector<3, T> const& c3)
    : col0(m.col0)
    , col1(m.col1)
    , col2(m.col2)
    , col3(c3)
{
}

template <typename T>
MATH_FUNC
inline matrix<4, 3, T>::matrix(T const data[12])
    : col0(&data[0])
    , col1(&data[3])
    , col2(&data[6])
    , col3(&data[9])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<4, 3, T>::matrix(matrix<4, 3, U> const& rhs)
    : col0(rhs.col0)
    , col1(rhs.col1)
    , col2(rhs.col2)
    , col3(rhs.col3)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<4, 3, T>& matrix<4, 3, T>::operator=(matrix<4, 3, U> const& rhs)
{
    col0 = rhs.col0;
    col1 = rhs.col1;
    col2 = rhs.col2;
    col3 = rhs.col3;
    return *this;
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<4, 3, T>::matrix(matrix<3, 3, U> const& rhs)
    : col0(rhs.col0)
    , col1(rhs.col1)
    , col2(rhs.col2)
    , col3(vector<3, T>(U(0.0)))
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<4, 3, T>& matrix<4, 3, T>::operator=(matrix<3, 3, U> const& rhs)
{
    col0 = rhs.col0;
    col1 = rhs.col1;
    col2 = rhs.col2;
    col3 = vector<3, T>(U(0.0));
    return *this;
}

template <typename T>
MATH_FUNC
inline T* matrix<4, 3, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* matrix<4, 3, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline vector<3, T>& matrix<4, 3, T>::operator()(size_t col)
{
    return *(reinterpret_cast<vector<3, T>*>(this) + col);
}

template <typename T>
MATH_FUNC
inline vector<3, T> const& matrix<4, 3, T>::operator()(size_t col) const
{
    return *(reinterpret_cast<vector<3, T> const*>(this) + col);
}

template <typename T>
MATH_FUNC
inline T& matrix<4, 3, T>::operator()(size_t row, size_t col)
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline T const& matrix<4, 3, T>::operator()(size_t row, size_t col) const
{
    return (operator()(col))[row];
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline vector<3, T> operator*(matrix<4, 3, T> const& m, vector<4, T> const& v)
{
    return vector<3, T>(
            m(0, 0) * v.x + m(1, 0) * v.y + m(2, 0) * v.z + m(3, 0) * v.w,
            m(0, 1) * v.x + m(1, 1) * v.y + m(2, 1) * v.z + m(3, 1) * v.w,
            m(0, 2) * v.x + m(1, 2) * v.y + m(2, 2) * v.z + m(3, 2) * v.w
            );
}

template <typename T>
MATH_FUNC
inline vector<4, T> operator*(vector<3, T> const& v, matrix<4, 3, T> const& m)
{
    return vector<4, T>(
            v.x * m(0, 0) + v.y * m(0, 1) + v.z * m(0, 2),
            v.x * m(1, 0) + v.y * m(1, 1) + v.z * m(1, 2),
            v.x * m(2, 0) + v.y * m(2, 1) + v.z * m(2, 2),
            v.x * m(3, 0) + v.y * m(3, 1) + v.z * m(3, 2)
            );
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
inline bool operator==(matrix<4, 3, T> const& a, matrix<4, 3, T> const& b)
{
    return a(0) == b(0) && a(1) == b(1) && a(2) == b(2) && a(3) == b(3);
}

template <typename T>
MATH_FUNC
inline bool operator!=(matrix<4, 3, T> const& a, matrix<4, 3, T> const& b)
{
    return !(a == b);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

// Return top-left 3x3 matrix
template <typename T>
MATH_FUNC
inline matrix<3, 3, T> top_left(matrix<4, 3, T> const& m)
{
    matrix<3, 3, T> result;

    for (unsigned y = 0; y < 3; ++y)
    {
        for (unsigned x = 0; x < 3; ++x)
        {
            result(x, y) = m(y, x);
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Misc.
//

template <typename M, typename T>
MATH_FUNC
matrix<4, 3, T> select(M const& m, matrix<4, 3, T> const& u, matrix<4, 3, T> const& v)
{
    return matrix<4, 3, T>( 
            select(m, u.col0, v.col0),
            select(m, u.col1, v.col1),
            select(m, u.col2, v.col2),
            select(m, u.col3, v.col3)
            );  
}

namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// unpack -------------------------------------------------

template <
    typename T,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type
    >
MATH_FUNC
inline array<matrix<4, 3, element_type_t<T>>, num_elements<T>::value> unpack(matrix<4, 3, T> const& m)
{
    using U = element_type_t<T>;

    U const* c0x = reinterpret_cast<U const*>(&m.col0.x);
    U const* c1x = reinterpret_cast<U const*>(&m.col1.x);
    U const* c2x = reinterpret_cast<U const*>(&m.col2.x);
    U const* c3x = reinterpret_cast<U const*>(&m.col3.x);

    U const* c0y = reinterpret_cast<U const*>(&m.col0.y);
    U const* c1y = reinterpret_cast<U const*>(&m.col1.y);
    U const* c2y = reinterpret_cast<U const*>(&m.col2.y);
    U const* c3y = reinterpret_cast<U const*>(&m.col3.y);

    U const* c0z = reinterpret_cast<U const*>(&m.col0.z);
    U const* c1z = reinterpret_cast<U const*>(&m.col1.z);
    U const* c2z = reinterpret_cast<U const*>(&m.col2.z);
    U const* c3z = reinterpret_cast<U const*>(&m.col3.z);

    array<matrix<4, 3, U>, num_elements<T>::value> result;

    for (int i = 0; i < num_elements<T>::value; ++i)
    {
        result[i].col0.x = c0x[i];
        result[i].col0.y = c0y[i];
        result[i].col0.z = c0z[i];

        result[i].col1.x = c1x[i];
        result[i].col1.y = c1y[i];
        result[i].col1.z = c1z[i];

        result[i].col2.x = c2x[i];
        result[i].col2.y = c2y[i];
        result[i].col2.z = c2z[i];

        result[i].col3.x = c3x[i];
        result[i].col3.y = c3y[i];
        result[i].col3.z = c3z[i];
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
