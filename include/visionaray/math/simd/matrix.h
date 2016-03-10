// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VISIONARAY_SIMD_MATRIX_H
#define VISIONARAY_SIMD_MATRIX_H 1

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
    matrix(
        column_type const& c0,
        column_type const& c1,
        column_type const& c2,
        column_type const& c3
        );

    explicit matrix(float const data[16]);

    template <typename U>
    explicit matrix(matrix<4, 4, U> const& rhs);

    template <typename U>
    matrix& operator=(matrix<4, 4, U> const& rhs);

    column_type& operator()(size_t col);
    column_type const& operator()(size_t col) const;

    static matrix identity();

};


//-------------------------------------------------------------------------------------------------
// Free function declarations
//

matrix<4, 4, simd::float4> operator*(
        matrix<4, 4, simd::float4> const& a,
        matrix<4, 4, simd::float4> const& b
        );

vector<4, simd::float4> operator*(
        matrix<4, 4, simd::float4> const& m,
        vector<4, simd::float4> const& v
        );

matrix<4, 4, simd::float4> transpose(matrix<4, 4, simd::float4> const& m);


} // MATH_NAMESPACE

#include "detail/sse/matrix4.inl"

#endif // VISIONARAY_SIMD_MATRIX_H
