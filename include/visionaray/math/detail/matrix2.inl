// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// matrix2 members
//

template <typename T>
MATH_FUNC
inline matrix<2, 2, T>::matrix(vector<2, T> const& c0, vector<2, T> const& c1)
    : col0(c0)
    , col1(c1)
{
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T>::matrix(T const& m00, T const& m10, T const& m01, T const& m11)
    : col0(m00, m10)
    , col1(m01, m11)
{
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T>::matrix(T const& m00, T const& m11)
    : col0(m00, T(0.0))
    , col1(T(0.0), m11)
{
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T>::matrix(T const data[4])
    : col0(&data[0])
    , col1(&data[2])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<2, 2, T>::matrix(matrix<2, 2, U> const& rhs)
    : col0(rhs.col0)
    , col1(rhs.col1)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<2, 2, T>& matrix<2, 2, T>::operator=(matrix<2, 2, U> const& rhs)
{
    col0 = rhs.col0;
    col1 = rhs.col1;
    return *this;
}

template <typename T>
MATH_FUNC
inline T* matrix<2, 2, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* matrix<2, 2, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline vector<2, T>& matrix<2, 2, T>::operator()(size_t col)
{
    return *(reinterpret_cast<vector<2, T>*>(this) + col);
}

template <typename T>
MATH_FUNC
inline vector<2, T> const& matrix<2, 2, T>::operator()(size_t col) const
{
    return *(reinterpret_cast<vector<2, T> const*>(this) + col);
}

template <typename T>
MATH_FUNC
inline T& matrix<2, 2, T>::operator()(size_t row, size_t col)
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline T const& matrix<2, 2, T>::operator()(size_t row, size_t col) const
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> matrix<2, 2, T>::identity()
{
    return matrix<2, 2, T>(
            T(1.0), T(0.0),
            T(0.0), T(1.0)
            );
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> matrix<2, 2, T>::scaling(vector<2, T> const& v)
{
    matrix<2, 2, T> s = identity();
    s(0, 0) = v.x;
    s(1, 1) = v.y;
    return s;
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> matrix<2, 2, T>::scaling(T const& x, T const& y)
{
    return scaling(vector<2, T>(x, y));
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> operator*(matrix<2, 2, T> const& a, matrix<2, 2, T> const& b)
{
    return matrix<2, 2, T>(
            a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0),
            a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0),
            a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1),
            a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1)
            );
}

template <typename T>
MATH_FUNC
inline vector<2, T> operator*(matrix<2, 2, T> const& m, vector<2, T> const& v)
{
    return vector<2, T>(
            m(0, 0) * v.x + m(0, 1) * v.y,
            m(1, 0) * v.x + m(1, 1) * v.y
            );
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> operator*(matrix<2, 2, T> const& m, T const& s)
{
    return matrix<2, 2, T>(
            m.col0 * s,
            m.col1 * s
            );
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> operator*(T const& s, matrix<2, 2, T> const& m)
{
    return matrix<2, 2, T>(
            s * m.col0,
            s * m.col1
            );
}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> inverse(matrix<2, 2, T> const& m)
{
    T a00 =  m(1, 1);
    T a10 = -m(1, 0);
    T a01 = -m(0, 1);
    T a11 =  m(0, 0);

    T det = det2(m(0, 0), m(0, 1), m(1, 0), m(1, 1));

    return matrix<2, 2, T>(a00 / det, a10 / det, a01 / det, a11 / det);
}

template <typename T>
MATH_FUNC
inline matrix<2, 2, T> transpose(matrix<2, 2, T> const& m)
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
