// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SAMPLER_COMMON_H
#define VSNRAY_SAMPLER_COMMON_H


#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>

#include "../forward.h"


namespace visionaray
{
namespace detail
{

template <size_t Dim, typename T>
vector<Dim, T> map_tex_coord(vector<Dim, T> const& coord, std::array<tex_address_mode, Dim> const& mode)
{
    vector<Dim, T> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        switch (mode[d])
        {

        case Wrap:
            result[d] = coord[d] - floor(coord[d]);
            break;

        case Clamp:
            // fall-through
        default:
            result[d] = clamp( coord[d], T(0.0), T(1.0) );
            break;

        }
    }

    return result;
}

template <typename T>
inline T point(T const* tex, ptrdiff_t idx)
{
    return tex[idx];
}


template <typename T>
inline simd::float4 point(T const* tex, simd::float4 idx)
{

    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], idx);
    return simd::float4
    (
        tex[indices[0]],
        tex[indices[1]],
        tex[indices[2]],
        tex[indices[3]]
    );

}

inline vector<3, simd::float4> point(vector<3, unorm<8>> const* tex, simd::float4 idx)
{
    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], idx);
    return vector<3, simd::float4>
    (
        simd::float4(tex[indices[0]].x, tex[indices[1]].x, tex[indices[2]].x, tex[indices[3]].x),
        simd::float4(tex[indices[0]].y, tex[indices[1]].y, tex[indices[2]].y, tex[indices[3]].y),
        simd::float4(tex[indices[0]].z, tex[indices[1]].z, tex[indices[2]].z, tex[indices[3]].z)
    );
}

inline vector<4, simd::float4> point(vector<4, float> const* tex, simd::float4 idx)
{

    // Special case: colors are AoS. Those can be obtained
    // without a context switch to GP registers by transposing
    // to SoA after memory lookup.

    simd::float4 iidx( idx * 4 );
    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], iidx);

    float const* tmp = reinterpret_cast<float const*>(tex);

    vector<4, simd::float4> colors
    (
        &tmp[0] + indices[0],
        &tmp[0] + indices[1],
        &tmp[0] + indices[2],
        &tmp[0] + indices[3]
    );

    colors = transpose(colors);
    return colors;

}


namespace bspline
{

// weight functions for Mitchell - Netravalli B-Spline with B = 1 and C = 0

template <typename FloatT>
struct w0_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (-(a * a * a) + 3.0 * a * a - 3.0 * a + 1.0) );
    }
};

template <typename FloatT>
struct w1_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (3.0 * a * a * a - 6.0 * a * a + 4.0) );
    }
};

template <typename FloatT>
struct w2_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (-3.0 * a * a * a + 3.0 * a * a + 3.0 * a + 1.0) );
    }
};

template <typename FloatT>
struct w3_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (a * a * a) );
    }
};

} // bspline

namespace cspline
{

// weight functions for Catmull - Rom Cardinal Spline

template <typename FloatT>
struct w0_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( -0.5 * a * a * a + a * a - 0.5 * a );
    }
};

template <typename FloatT>
struct w1_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( 1.5 * a * a * a - 2.5 * a * a + 1.0 );
    }
};

template <typename FloatT>
struct w2_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( -1.5 * a * a * a + 2.0 * a * a + 0.5 * a );
    }
};

template <typename FloatT>
struct w3_func
{
    inline FloatT operator()( FloatT a )
    {
        return FloatT( 0.5 * a * a * a - 0.5 * a * a );
    }
};

} // cspline

// helper functions for cubic interpolation
template <typename FloatT>
inline FloatT g0( FloatT x )
{
    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    return w0(x) + w1(x);
}

template <typename FloatT>
inline FloatT g1( FloatT x )
{
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;
    return w2(x) + w3(x);
}

template <typename FloatT>
inline FloatT h0( FloatT x )
{
    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    return ((floor( x ) - FloatT(1.0) + w1(x)) / (w0(x) + w1(x))) + x;
}

template <typename FloatT>
inline FloatT h1( FloatT x )
{
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;
    return ((floor( x ) + FloatT(1.0) + w3(x)) / (w2(x) + w3(x))) - x;
}


} // detail
} // visionaray


#endif // VSNRAY_SAMPLER_COMMON_H


