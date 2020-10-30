// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_MATRIX_H
#define VSNRAY_MATH_MATRIX_H 1

#include <cstddef>

#include "forward.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T>
class matrix<2, 2, T>
{
public:

    using column_type = vector<2, T>;

public:

    column_type col0;
    column_type col1;

public:

    matrix() = default;

    MATH_FUNC matrix(column_type const& c0, column_type const& c1);

    MATH_FUNC matrix(T const& m00, T const& m10, T const& m01, T const& m11);

    MATH_FUNC matrix(T const& m00, T const& m11);

    MATH_FUNC
    explicit matrix(T const data[4]);

    template <typename U>
    MATH_FUNC
    explicit matrix(matrix<2, 2, U> const& rhs);

    template <typename U>
    MATH_FUNC
    matrix& operator=(matrix<2, 2, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC column_type& operator()(size_t col);
    MATH_FUNC column_type const& operator()(size_t col) const;

    MATH_FUNC T& operator()(size_t row, size_t col);
    MATH_FUNC T const& operator()(size_t row, size_t col) const;

    // Construct identity matrix
    MATH_FUNC static matrix identity();

    // TODO:
    // Construct rotation matrix from axis and angle
    //MATH_FUNC static matrix rotation(vector<2, T> const& axis, T const& angle);

    // Construct scaling matrix from vector
    MATH_FUNC static matrix scaling(vector<2, T> const& v);

    // Construct scaling matrix from x,y,z
    MATH_FUNC static matrix scaling(T const& x, T const& y);

};

template <typename T>
class matrix<3, 3, T>
{
public:

    using column_type = vector<3, T>;

public:

    column_type col0;
    column_type col1;
    column_type col2;

public:

    matrix() = default;

    MATH_FUNC matrix(
            column_type const& c0,
            column_type const& c1,
            column_type const& c2
            );

    MATH_FUNC matrix(
            T const& m00, T const& m10, T const& m20,
            T const& m01, T const& m11, T const& m21,
            T const& m02, T const& m12, T const& m22
            );

    MATH_FUNC matrix(T const& m00, T const& m11, T const& m22);

    MATH_FUNC
    explicit matrix(T const data[9]);

    template <typename U>
    MATH_FUNC
    explicit matrix(matrix<3, 3, U> const& rhs);

    template <typename U>
    MATH_FUNC
    matrix& operator=(matrix<3, 3, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC column_type& operator()(size_t col);
    MATH_FUNC column_type const& operator()(size_t col) const;

    MATH_FUNC T& operator()(size_t row, size_t col);
    MATH_FUNC T const& operator()(size_t row, size_t col) const;

    // Construct identity matrix
    MATH_FUNC static matrix identity();

    // Construct rotation matrix from axis and angle
    MATH_FUNC static matrix rotation(vector<3, T> const& axis, T const& angle);

    // Construct scaling matrix from vector
    MATH_FUNC static matrix scaling(vector<3, T> const& v);

    // Construct scaling matrix from x,y,z
    MATH_FUNC static matrix scaling(T const& x, T const& y, T const& z);

};

template <typename T>
class matrix<4, 3, T>
{
public:

    using column_type = vector<3, T>;

public:

    column_type col0;
    column_type col1;
    column_type col2;
    column_type col3;

public:

    matrix() = default;

    MATH_FUNC matrix(
            column_type const& c0,
            column_type const& c1,
            column_type const& c2,
            column_type const& c3
            );

    MATH_FUNC matrix(
            T const& m00, T const& m10, T const& m20,
            T const& m01, T const& m11, T const& m21,
            T const& m02, T const& m12, T const& m22,
            T const& m03, T const& m13, T const& m23
            );

    MATH_FUNC
    explicit matrix(T const data[12]);

    template <typename U>
    MATH_FUNC
    explicit matrix(matrix<4, 3, U> const& rhs);

    template <typename U>
    MATH_FUNC
    matrix& operator=(matrix<4, 3, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC column_type& operator()(size_t col);
    MATH_FUNC column_type const& operator()(size_t col) const;

    MATH_FUNC T& operator()(size_t row, size_t col);
    MATH_FUNC T const& operator()(size_t row, size_t col) const;

};

template <typename T>
class matrix<4, 4, T>
{
public:

    using column_type = vector<4, T>;

public:

    column_type col0;
    column_type col1;
    column_type col2;
    column_type col3;

public:

    matrix() = default;

    MATH_FUNC matrix(
            column_type const& c0,
            column_type const& c1,
            column_type const& c2,
            column_type const& c3
            );

    MATH_FUNC matrix(
            T const& m00, T const& m10, T const& m20, T const& m30,
            T const& m01, T const& m11, T const& m21, T const& m31,
            T const& m02, T const& m12, T const& m22, T const& m32,
            T const& m03, T const& m13, T const& m23, T const& m33
            );

    MATH_FUNC matrix(T const& m00, T const& m11, T const& m22, T const& m33);

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

    // Construct identity matrix
    MATH_FUNC static matrix identity();

    // Construct rotation matrix from axis and angle
    MATH_FUNC static matrix rotation(vector<3, T> const& axis, T const& angle);

    // Construct scaling matrix from vector
    MATH_FUNC static matrix scaling(vector<3, T> const& v);

    // Construct scaling matrix x,y,z
    MATH_FUNC static matrix scaling(T const& x, T const& y, T const& z);

    // Construct translation matrix from vector
    MATH_FUNC static matrix translation(vector<3, T> const& v);

    // Construct translation matrix from x,y,z
    MATH_FUNC static matrix translation(T const& x, T const& y, T const& z);

};

template <size_t N, size_t M, typename T>
class matrix
{
public:

    using column_type = vector<N, T>;

public:

    column_type cols[M];

    // Construct identity matrix (requires N == M!)
    MATH_FUNC static matrix identity();

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC column_type& operator()(size_t col);
    MATH_FUNC column_type const& operator()(size_t col) const;

    MATH_FUNC T& operator()(size_t row, size_t col);
    MATH_FUNC T const& operator()(size_t row, size_t col) const;

};

} // MATH_NAMESPACE

#include "detail/matrix.inl"
#include "detail/matrix2.inl"
#include "detail/matrix3.inl"
#include "detail/matrix4x3.inl"
#include "detail/matrix4.inl"

#endif // VSNRAY_MATH_MATRIX_H
