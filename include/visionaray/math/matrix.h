// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_MATRIX_H
#define VSNRAY_MATH_MATRIX_H

#include <cassert>

#include "vector.h"


namespace MATH_NAMESPACE
{


template <typename T>
class matrix<4, 4, T>
{
public:

    typedef vector<4, T> column_type;

    column_type col0;
    column_type col1;
    column_type col2;
    column_type col3;

    MATH_FUNC
    matrix();

    MATH_FUNC
    matrix(column_type const& c0,
           column_type const& c1,
           column_type const& c2,
           column_type const& c3);

    MATH_FUNC
    matrix(T const& m00, T const& m10, T const& m20, T const& m30,
           T const& m01, T const& m11, T const& m21, T const& m31,
           T const& m02, T const& m12, T const& m22, T const& m32,
           T const& m03, T const& m13, T const& m23, T const& m33);

    MATH_FUNC
    matrix(T const& m00, T const& m11, T const& m22, T const& m33);

    MATH_FUNC
    explicit matrix(T const data[16]);

    template <typename U>
    MATH_FUNC
    explicit matrix(matrix<4, 4, U> const& rhs);

    template <typename U>
    MATH_FUNC
    matrix& operator=(matrix<4, 4, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC column_type& operator()(size_t col);
    MATH_FUNC column_type const& operator()(size_t col) const;

    MATH_FUNC T& operator()(size_t row, size_t col);
    MATH_FUNC T const& operator()(size_t row, size_t col) const;

    MATH_FUNC static matrix identity();

};


} // MATH_NAMESPACE


#include "detail/matrix4.inl"


#endif // VSNRAY_MATH_MATRIX_H


