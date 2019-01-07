// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// matrix3 members
//

template <typename T>
MATH_FUNC
inline matrix<3, 3, T>::matrix(
        vector<3, T> const& c0,
        vector<3, T> const& c1,
        vector<3, T> const& c2
        )
    : col0(c0)
    , col1(c1)
    , col2(c2)
{
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T>::matrix(
        T const& m00, T const& m10, T const& m20,
        T const& m01, T const& m11, T const& m21,
        T const& m02, T const& m12, T const& m22
        )
    : col0(m00, m10, m20)
    , col1(m01, m11, m21)
    , col2(m02, m12, m22)
{
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T>::matrix(T const& m00, T const& m11, T const& m22)
    : col0(m00,    T(0.0), T(0.0))
    , col1(T(0.0), m11,    T(0.0))
    , col2(T(0.0), T(0.0), m22   )
{
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T>::matrix(T const data[9])
    : col0(&data[0])
    , col1(&data[3])
    , col2(&data[6])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<3, 3, T>::matrix(matrix<3, 3, U> const& rhs)
    : col0(rhs.col0)
    , col1(rhs.col1)
    , col2(rhs.col2)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<3, 3, T>& matrix<3, 3, T>::operator=(matrix<3, 3, U> const& rhs)
{
    col0 = rhs.col0;
    col1 = rhs.col1;
    col2 = rhs.col2;
    return *this;
}

template <typename T>
MATH_FUNC
inline T* matrix<3, 3, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* matrix<3, 3, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline vector<3, T>& matrix<3, 3, T>::operator()(size_t col)
{
    return *(reinterpret_cast<vector<3, T>*>(this) + col);
}

template <typename T>
MATH_FUNC
inline vector<3, T> const& matrix<3, 3, T>::operator()(size_t col) const
{
    return *(reinterpret_cast<vector<3, T> const*>(this) + col);
}

template <typename T>
MATH_FUNC
inline T& matrix<3, 3, T>::operator()(size_t row, size_t col)
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline T const& matrix<3, 3, T>::operator()(size_t row, size_t col) const
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> matrix<3, 3, T>::identity()
{
    return matrix<3, 3, T>(
            T(1.0), T(0.0), T(0.0),
            T(0.0), T(1.0), T(0.0),
            T(0.0), T(0.0), T(1.0)
            );
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> matrix<3, 3, T>::rotation(vector<3, T> const& axis, T const& angle)
{
    auto v = normalize(axis);
    auto s = sin(angle);
    auto c = cos(angle);

    return matrix<3, 3, T>(
            v.x * v.x * (T(1.0) - c) + c,
            v.x * v.y * (T(1.0) - c) + s * v.z,
            v.x * v.z * (T(1.0) - c) - s * v.y,

            v.y * v.x * (T(1.0) - c) - s * v.z,
            v.y * v.y * (T(1.0) - c) + c,
            v.y * v.z * (T(1.0) - c) + s * v.x,

            v.z * v.x * (T(1.0) - c) + s * v.y,
            v.z * v.y * (T(1.0) - c) - s * v.x,
            v.z * v.z * (T(1.0) - c) + c
            );
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> matrix<3, 3, T>::scaling(vector<3, T> const& v)
{
    matrix<3, 3, T> s = identity();
    s(0, 0) = v.x;
    s(1, 1) = v.y;
    s(2, 2) = v.z;
    return s;
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> matrix<3, 3, T>::scaling(T const& x, T const& y, T const& z)
{
    return scaling(vector<3, T>(x, y, z));
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> operator*(matrix<3, 3, T> const& a, matrix<3, 3, T> const& b)
{
    return matrix<3, 3, T>(
            a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0),
            a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0),
            a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0),
            a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1),
            a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1),
            a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1),
            a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2),
            a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2),
            a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2)
            );
}

template <typename T>
MATH_FUNC
inline vector<3, T> operator*(matrix<3, 3, T> const& m, vector<3, T> const& v)
{
    return vector<3, T>(
            m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z,
            m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z,
            m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z
            );
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> operator*(matrix<3, 3, T> const& m, T const& s)
{
    return matrix<3, 3, T>(
            m.col0 * s,
            m.col1 * s,
            m.col2 * s
            );
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> operator*(T const& s, matrix<3, 3, T> const& m)
{
    return matrix<3, 3, T>(
            s * m.col0,
            s * m.col1,
            s * m.col2
            );
}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> inverse(matrix<3, 3, T> const& m)
{
    T a00 = det2(m(1, 1), m(1, 2), m(2, 1), m(2, 2));
    T a01 = det2(m(1, 0), m(1, 2), m(2, 0), m(2, 2));
    T a02 = det2(m(1, 0), m(1, 1), m(2, 0), m(2, 1));
    T a10 = det2(m(0, 1), m(0, 2), m(2, 1), m(2, 2));
    T a11 = det2(m(0, 0), m(0, 2), m(2, 0), m(2, 2));
    T a12 = det2(m(0, 0), m(0, 1), m(2, 0), m(2, 1));
    T a20 = det2(m(0, 1), m(0, 2), m(1, 1), m(1, 2));
    T a21 = det2(m(0, 0), m(0, 2), m(1, 0), m(1, 2));
    T a22 = det2(m(0, 0), m(0, 1), m(1, 0), m(1, 1));

    T det = m(0, 0) * a00 - m(0, 1) * a01 + m(0, 2) * a02;

    return matrix<3, 3, T>(
            +a00 / det, -a01 / det, +a02 / det,
            -a10 / det, +a11 / det, -a12 / det,
            +a20 / det, -a21 / det, +a22 / det
            );
}

template <typename T>
MATH_FUNC
inline T trace(matrix<3, 3, T> const& m)
{
    return m(0, 0) + m(1, 1) + m(2, 2);
}

template <typename T>
MATH_FUNC
inline matrix<3, 3, T> transpose(matrix<3, 3, T> const& m)
{

    matrix<3, 3, T> result;

    for (size_t y = 0; y < 3; ++y)
    {
        for (size_t x = 0; x < 3; ++x)
        {
            result(x, y) = m(y, x);
        }
    }

    return result;

}

// Return top-left 2x2 matrix
template <typename T>
MATH_FUNC
inline matrix<2, 2, T> top_left(matrix<3, 3, T> const& m)
{
    matrix<2, 2, T> result;

    for (size_t y = 0; y < 2; ++y)
    {
        for (size_t x = 0; x < 2; ++x)
        {
            result(x, y) = m(y, x);
        }
    }

    return result;
}

} // MATH_NAMESPACE
