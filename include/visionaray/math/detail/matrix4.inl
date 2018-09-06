// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// matrix4 members
//

template <typename T>
MATH_FUNC
inline matrix<4, 4, T>::matrix(
        vector<4, T> const& c0,
        vector<4, T> const& c1,
        vector<4, T> const& c2,
        vector<4, T> const& c3
        )
    : col0(c0)
    , col1(c1)
    , col2(c2)
    , col3(c3)
{
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T>::matrix(
        T const& m00, T const& m10, T const& m20, T const& m30,
        T const& m01, T const& m11, T const& m21, T const& m31,
        T const& m02, T const& m12, T const& m22, T const& m32,
        T const& m03, T const& m13, T const& m23, T const& m33
        )
    : col0(m00, m10, m20, m30)
    , col1(m01, m11, m21, m31)
    , col2(m02, m12, m22, m32)
    , col3(m03, m13, m23, m33)
{
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T>::matrix(T const& m00, T const& m11, T const& m22, T const& m33)
    : col0(m00,    T(0.0), T(0.0), T(0.0))
    , col1(T(0.0), m11,    T(0.0), T(0.0))
    , col2(T(0.0), T(0.0), m22,    T(0.0))
    , col3(T(0.0), T(0.0), T(0.0), m33   )
{
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T>::matrix(T const data[16])
    : col0(&data[ 0])
    , col1(&data[ 4])
    , col2(&data[ 8])
    , col3(&data[12])
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<4, 4, T>::matrix(matrix<4, 4, U> const& rhs)
    : col0(rhs.col0)
    , col1(rhs.col1)
    , col2(rhs.col2)
    , col3(rhs.col3)
{
}

template <typename T>
template <typename U>
MATH_FUNC
inline matrix<4, 4, T>& matrix<4, 4, T>::operator=(matrix<4, 4, U> const& rhs)
{
    col0 = rhs.col0;
    col1 = rhs.col1;
    col2 = rhs.col2;
    col3 = rhs.col3;
    return *this;
}

template <typename T>
MATH_FUNC
inline T* matrix<4, 4, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* matrix<4, 4, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline vector<4, T>& matrix<4, 4, T>::operator()(size_t col)
{
    return *(reinterpret_cast<vector<4, T>*>(this) + col);
}

template <typename T>
MATH_FUNC
inline vector<4, T> const& matrix<4, 4, T>::operator()(size_t col) const
{
    return *(reinterpret_cast<vector<4, T> const*>(this) + col);
}

template <typename T>
MATH_FUNC
inline T& matrix<4, 4, T>::operator()(size_t row, size_t col)
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline T const& matrix<4, 4, T>::operator()(size_t row, size_t col) const
{
    return (operator()(col))[row];
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> matrix<4, 4, T>::identity()
{
    return matrix<4, 4, T>(
            T(1.0), T(0.0), T(0.0), T(0.0),
            T(0.0), T(1.0), T(0.0), T(0.0),
            T(0.0), T(0.0), T(1.0), T(0.0),
            T(0.0), T(0.0), T(0.0), T(1.0)
            );
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> matrix<4, 4, T>::rotation(vector<3, T> const& axis, T const& angle)
{
    auto v = normalize(axis);
    auto s = sin(angle);
    auto c = cos(angle);

    return matrix<4, 4, T>(
            v.x * v.x * (T(1.0) - c) + c,
            v.x * v.y * (T(1.0) - c) + s * v.z,
            v.x * v.z * (T(1.0) - c) - s * v.y,
            T(0.0),
            v.y * v.x * (T(1.0) - c) - s * v.z,
            v.y * v.y * (T(1.0) - c) + c,
            v.y * v.z * (T(1.0) - c) + s * v.x,
            T(0.0),
            v.z * v.x * (T(1.0) - c) + s * v.y,
            v.z * v.y * (T(1.0) - c) - s * v.x,
            v.z * v.z * (T(1.0) - c) + c,
            T(0.0),
            T(0.0),
            T(0.0),
            T(0.0),
            T(1.0)
            );
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> matrix<4, 4, T>::scaling(vector<3, T> const& v)
{
    matrix<4, 4, T> s = identity();
    s(0, 0) = v.x;
    s(1, 1) = v.y;
    s(2, 2) = v.z;
    return s;
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> matrix<4, 4, T>::scaling(T const& x, T const& y, T const& z)
{
    return scaling(vector<3, T>(x, y, z));
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> matrix<4, 4, T>::translation(vector<3, T> const& v)
{
    matrix<4, 4, T> s = identity();
    s(0, 3) = v.x;
    s(1, 3) = v.y;
    s(2, 3) = v.z;
    return s;
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> matrix<4, 4, T>::translation(T const& x, T const& y, T const& z)
{
    return translation(vector<3, T>(x, y, z));
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> operator*(matrix<4, 4, T> const& a, matrix<4, 4, T> const& b)
{

    return matrix<4, 4, T>(
            a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0) + a(0, 3) * b(3, 0),
            a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0) + a(1, 3) * b(3, 0),
            a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0) + a(2, 3) * b(3, 0),
            a(3, 0) * b(0, 0) + a(3, 1) * b(1, 0) + a(3, 2) * b(2, 0) + a(3, 3) * b(3, 0),
            a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1) + a(0, 3) * b(3, 1),
            a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1) + a(1, 3) * b(3, 1),
            a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1) + a(2, 3) * b(3, 1),
            a(3, 0) * b(0, 1) + a(3, 1) * b(1, 1) + a(3, 2) * b(2, 1) + a(3, 3) * b(3, 1),
            a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2) + a(0, 3) * b(3, 2),
            a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2) + a(1, 3) * b(3, 2),
            a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2) + a(2, 3) * b(3, 2),
            a(3, 0) * b(0, 2) + a(3, 1) * b(1, 2) + a(3, 2) * b(2, 2) + a(3, 3) * b(3, 2),
            a(0, 0) * b(0, 3) + a(0, 1) * b(1, 3) + a(0, 2) * b(2, 3) + a(0, 3) * b(3, 3),
            a(1, 0) * b(0, 3) + a(1, 1) * b(1, 3) + a(1, 2) * b(2, 3) + a(1, 3) * b(3, 3),
            a(2, 0) * b(0, 3) + a(2, 1) * b(1, 3) + a(2, 2) * b(2, 3) + a(2, 3) * b(3, 3),
            a(3, 0) * b(0, 3) + a(3, 1) * b(1, 3) + a(3, 2) * b(2, 3) + a(3, 3) * b(3, 3)
            );

}

template <typename T>
MATH_FUNC
inline vector<4, T> operator*(matrix<4, 4, T> const& m, vector<4, T> const& v)
{

    return vector<4, T>(
            m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z + m(0, 3) * v.w,
            m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z + m(1, 3) * v.w,
            m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z + m(2, 3) * v.w,
            m(3, 0) * v.x + m(3, 1) * v.y + m(3, 2) * v.z + m(3, 3) * v.w
            );

}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
inline bool operator==(matrix<4, 4, T> const& a, matrix<4, 4, T> const& b)
{
    return a(0) == b(0) && a(1) == b(1) && a(2) == b(2) && a(3) == b(3);
}

template <typename T>
MATH_FUNC
inline bool operator!=(matrix<4, 4, T> const& a, matrix<4, 4, T> const& b)
{
    return !(a == b);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> inverse(matrix<4, 4, T> const& m)
{
    T s0 = det2(m(0, 0), m(0, 1), m(1, 0), m(1, 1));
    T s1 = det2(m(0, 0), m(0, 2), m(1, 0), m(1, 2));
    T s2 = det2(m(0, 0), m(0, 3), m(1, 0), m(1, 3));
    T s3 = det2(m(0, 1), m(0, 2), m(1, 1), m(1, 2));
    T s4 = det2(m(0, 1), m(0, 3), m(1, 1), m(1, 3));
    T s5 = det2(m(0, 2), m(0, 3), m(1, 2), m(1, 3));
    T c5 = det2(m(2, 2), m(2, 3), m(3, 2), m(3, 3));
    T c4 = det2(m(2, 1), m(2, 3), m(3, 1), m(3, 3));
    T c3 = det2(m(2, 1), m(2, 2), m(3, 1), m(3, 2));
    T c2 = det2(m(2, 0), m(2, 3), m(3, 0), m(3, 3));
    T c1 = det2(m(2, 0), m(2, 2), m(3, 0), m(3, 2));
    T c0 = det2(m(2, 0), m(2, 1), m(3, 0), m(3, 1));

    T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    return matrix<4, 4, T>(
            (+ m(1, 1) * c5 - m(1, 2) * c4 + m(1, 3) * c3) / det,
            (- m(1, 0) * c5 + m(1, 2) * c2 + m(1, 3) * c1) / det,
            (+ m(1, 0) * c4 - m(1, 1) * c2 + m(1, 3) * c0) / det,
            (- m(1, 0) * c3 + m(1, 1) * c1 + m(1, 2) * c0) / det,
            (- m(0, 1) * c5 + m(0, 2) * c4 - m(0, 3) * c3) / det,
            (+ m(0, 0) * c5 - m(0, 2) * c2 + m(0, 3) * c1) / det,
            (- m(0, 0) * c4 + m(0, 1) * c2 - m(0, 3) * c0) / det,
            (+ m(0, 0) * c3 - m(0, 1) * c1 + m(0, 2) * c0) / det,
            (+ m(3, 1) * s5 - m(3, 2) * s4 + m(3, 3) * s3) / det,
            (- m(3, 0) * s5 + m(3, 2) * s2 - m(3, 3) * s1) / det,
            (+ m(3, 0) * s4 - m(3, 1) * s2 + m(3, 3) * s0) / det,
            (- m(3, 0) * s3 + m(3, 1) * s1 - m(3, 2) * s0) / det,
            (- m(2, 1) * s5 + m(2, 2) * s4 - m(2, 3) * s3) / det,
            (+ m(2, 0) * s5 - m(2, 2) * s2 + m(2, 3) * s1) / det,
            (- m(2, 0) * s4 + m(2, 1) * s2 - m(2, 3) * s0) / det,
            (+ m(2, 0) * s3 - m(2, 1) * s1 + m(2, 2) * s0) / det
            );
}

template <typename T>
MATH_FUNC
inline T trace(matrix<4, 4, T> const& m)
{
    return m(0, 0), + m(1, 1) + m(2, 2) + m(3, 3);
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> transpose(matrix<4, 4, T> const& m)
{

    matrix<4, 4, T> result;

    for (size_t y = 0; y < 4; ++y)
    {
        for (size_t x = 0; x < 4; ++x)
        {
            result(x, y) = m(y, x);
        }
    }

    return result;

}

// Return top-left 3x3 matrix
template <typename T>
MATH_FUNC
inline matrix<3, 3, T> top_left(matrix<4, 4, T> const& m)
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


//--------------------------------------------------------------------------------------------------
// Transforms
//

// get transforms

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> get_scaling(matrix<4, 4, T> const& m)
{
    auto s = matrix<4, 4, T>::identity();
    s(0, 0) = length(m.col0);
    s(1, 1) = length(m.col1);
    s(2, 2) = length(m.col2);
    return s;
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> get_translation(matrix<4, 4, T> const& m)
{
    auto t = matrix<4, 4, T>::identity();
    t(0, 3) = m(0, 3);
    t(1, 3) = m(1, 3);
    t(2, 3) = m(2, 3);
    return t;
}


// convenience functions to apply transforms

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> rotate(matrix<4, 4, T> const& m, vector<3, T> const& axis, T const& angle)
{
    return m * matrix<4, 4, T>::rotation(axis, angle);
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> scale(matrix<4, 4, T> const& m, vector<3, T> const& v)
{
    return m * matrix<4, 4, T>::scaling(v);
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> translate(matrix<4, 4, T> const& m, vector<3, T> const& v)
{
    return m * matrix<4, 4, T>::translation(v);
}


//-------------------------------------------------------------------------------------------------
// Misc.
//

template <typename M, typename T>
MATH_FUNC
matrix<4, 4, T> select(M const& m, matrix<4, 4, T> const& u, matrix<4, 4, T> const& v)
{
    return matrix<4, 4, T>( 
            select(m, u.col0, v.col0),
            select(m, u.col1, v.col1),
            select(m, u.col2, v.col2),
            select(m, u.col3, v.col3)
            );  
}

} // MATH_NAMESPACE
