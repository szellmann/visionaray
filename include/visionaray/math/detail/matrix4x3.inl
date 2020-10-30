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

} // MATH_NAMESPACE
