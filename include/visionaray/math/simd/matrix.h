// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_MATRIX_H
#define VSNRAY_MATH_SIMD_MATRIX_H 1

#include <cstddef>

#include "intrinsics.h"
#include "neon.h"
#include "sse.h"

namespace MATH_NAMESPACE
{

template <>
class matrix<4, 4, simd::float4>
{
public:

    typedef simd::float4 column_type;

    column_type col0;
    column_type col1;
    column_type col2;
    column_type col3;


    matrix() = default;

    MATH_FUNC
    matrix(
        column_type const& c0,
        column_type const& c1,
        column_type const& c2,
        column_type const& c3
        );

    MATH_FUNC
    explicit matrix(float const data[16]);

    template <typename U>
    MATH_FUNC
    explicit matrix(matrix<4, 4, U> const& rhs);

    template <typename U>
    MATH_FUNC
    matrix& operator=(matrix<4, 4, U> const& rhs);

    MATH_FUNC
    column_type& operator()(size_t col);

    MATH_FUNC
    column_type const& operator()(size_t col) const;

    MATH_FUNC
    static matrix identity();

};


//-------------------------------------------------------------------------------------------------
// Free function declarations
//

MATH_FUNC
matrix<4, 4, simd::float4> operator*(
        matrix<4, 4, simd::float4> const& a,
        matrix<4, 4, simd::float4> const& b
        );

MATH_FUNC
vector<4, simd::float4> operator*(
        matrix<4, 4, simd::float4> const& m,
        vector<4, simd::float4> const& v
        );

MATH_FUNC
matrix<4, 4, simd::float4> transpose(matrix<4, 4, simd::float4> const& m);


} // MATH_NAMESPACE

#include "detail/matrix4.inl"

#endif // VSNRAY_MATH_SIMD_MATRIX_H
