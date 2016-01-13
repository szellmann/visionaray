// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../../vector.h"


namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// matrix4 members
//

VSNRAY_FORCE_INLINE matrix<4, 4, simd::float4>::matrix
(
    simd::float4 const& c0,
    simd::float4 const& c1,
    simd::float4 const& c2,
    simd::float4 const& c3
)
    : col0(c0)
    , col1(c1)
    , col2(c2)
    , col3(c3)
{
}

VSNRAY_FORCE_INLINE matrix<4, 4, simd::float4>::matrix(float const data[16])
    : col0(&data[ 0])
    , col1(&data[ 4])
    , col2(&data[ 8])
    , col3(&data[12])
{
}

template <typename U>
inline matrix<4, 4, simd::float4>::matrix(matrix<4, 4, U> const& rhs)
    : col0( simd::float4(rhs.col0.x, rhs.col0.y, rhs.col0.z, rhs.col0.w) )
    , col1( simd::float4(rhs.col1.x, rhs.col1.y, rhs.col1.z, rhs.col1.w) )
    , col2( simd::float4(rhs.col2.x, rhs.col2.y, rhs.col2.z, rhs.col2.w) )
    , col3( simd::float4(rhs.col3.x, rhs.col3.y, rhs.col3.z, rhs.col3.w) )
{
}

template <typename U>
inline matrix<4, 4, simd::float4>& matrix<4, 4, simd::float4>::operator=(matrix<4, 4, U> const& rhs)
{
    col0 = simd::float4(rhs.col0.x, rhs.col0.y, rhs.col0.z, rhs.col0.w);
    col1 = simd::float4(rhs.col1.x, rhs.col1.y, rhs.col1.z, rhs.col1.w);
    col2 = simd::float4(rhs.col2.x, rhs.col2.y, rhs.col2.z, rhs.col2.w);
    col3 = simd::float4(rhs.col3.x, rhs.col3.y, rhs.col3.z, rhs.col3.w);
    return *this;
}

VSNRAY_FORCE_INLINE simd::float4& matrix<4, 4, simd::float4>::operator()(size_t col)
{
    return *(reinterpret_cast<simd::float4*>(this) + col);
}

VSNRAY_FORCE_INLINE simd::float4 const& matrix<4, 4, simd::float4>::operator()(size_t col) const
{
    return *(reinterpret_cast<simd::float4 const*>(this) + col);
}

inline matrix<4, 4, simd::float4> matrix<4, 4, simd::float4>::identity()
{
    return matrix<4, 4, simd::float4>(
            simd::float4(1.0f, 0.0f, 0.0f, 0.0f),
            simd::float4(0.0f, 1.0f, 0.0f, 0.0f),
            simd::float4(0.0f, 0.0f, 1.0f, 0.0f),
            simd::float4(0.0f, 0.0f, 0.0f, 1.0f)
            );
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

VSNRAY_FORCE_INLINE matrix<4, 4, simd::float4> operator*(
        matrix<4, 4, simd::float4> const& a,
        matrix<4, 4, simd::float4> const& b
        )
{

    using simd::shuffle;

    matrix<4, 4, simd::float4> result;

    for (size_t i = 0; i < 4; ++i)
    {
        result(i) = a(0) * shuffle<0, 0, 0, 0>( b(i) )
                  + a(1) * shuffle<1, 1, 1, 1>( b(i) )
                  + a(2) * shuffle<2, 2, 2, 2>( b(i) )
                  + a(3) * shuffle<3, 3, 3, 3>( b(i) );
    }

    return result;

}

VSNRAY_FORCE_INLINE vector<4, simd::float4> operator*(
        matrix<4, 4, simd::float4> const& m,
        vector<4, simd::float4> const& v
        )
{

    matrix<4, 4, simd::float4> tmp(v.x, v.y, v.z, v.w);
    matrix<4, 4, simd::float4> res = tmp * transpose(m);
    return vector<4, simd::float4>( res.col0, res.col1, res.col2, res.col3 );

}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

VSNRAY_FORCE_INLINE matrix<4, 4, simd::float4> transpose(matrix<4, 4, simd::float4> const& m)
{

    using simd::float4;

    float4 tmp0 = _mm_unpacklo_ps( m(0), m(1) );
    float4 tmp1 = _mm_unpacklo_ps( m(2), m(3) );
    float4 tmp2 = _mm_unpackhi_ps( m(0), m(1) );
    float4 tmp3 = _mm_unpackhi_ps( m(2), m(3) );

    return matrix<4, 4, simd::float4>(
            _mm_movelh_ps(tmp0, tmp1),
            _mm_movehl_ps(tmp1, tmp0),
            _mm_movelh_ps(tmp2, tmp3),
            _mm_movehl_ps(tmp3, tmp2)
            );

}

} // MATH_NAMESPACE
