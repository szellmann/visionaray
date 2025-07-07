// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_FILTER_COMMON_H
#define VSNRAY_TEXTURE_DETAIL_FILTER_COMMON_H 1

#include <visionaray/math/detail/math.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Weight functions for higher order texture interpolation
//

namespace bspline
{

// weight functions for Mitchell - Netravali B-Spline with B = 1 and C = 0

struct w0_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return (T(1.0) / T(6.0)) * (-(a * a * a) + T(3.0) * a * a - T(3.0) * a + T(1.0));
    }
};

struct w1_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return (T(1.0) / T(6.0)) * (T(3.0) * a * a * a - T(6.0) * a * a + T(4.0));
    }
};

struct w2_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return (T(1.0) / T(6.0)) * (T(-3.0) * a * a * a + T(3.0) * a * a + T(3.0) * a + T(1.0));
    }
};

struct w3_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return (T(1.0) / T(6.0)) * (a * a * a);
    }
};

} // bspline

namespace cspline
{

// weight functions for Catmull - Rom Cardinal Spline

struct w0_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return T(-0.5) * a * a * a + a * a - T(0.5) * a;
    }
};

struct w1_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return T(1.5) * a * a * a - T(2.5) * a * a + T(1.0);
    }
};

struct w2_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return T(-1.5) * a * a * a + T(2.0) * a * a + T(0.5) * a;
    }
};

struct w3_func
{
    template <typename T>
    VSNRAY_FUNC
    inline T operator()(T const& a)
    {
        return T(0.5) * a * a * a - T(0.5) * a * a;
    }
};

} // cspline

// helper functions for cubic interpolation
template <typename T>
VSNRAY_FUNC
inline T g0(T const& x)
{
    bspline::w0_func w0;
    bspline::w1_func w1;
    return w0(x) + w1(x);
}

template <typename T>
VSNRAY_FUNC
inline T g1(T const& x)
{
    bspline::w2_func w2;
    bspline::w3_func w3;
    return w2(x) + w3(x);
}

template <typename T>
VSNRAY_FUNC
inline T h0(T const& x)
{
    bspline::w0_func w0;
    bspline::w1_func w1;
    return ((floor( x ) - T(1.0) + w1(x)) / (w0(x) + w1(x))) + x;
}

template <typename T>
VSNRAY_FUNC
inline T h1(T const& x)
{
    bspline::w2_func w2;
    bspline::w3_func w3;
    return ((floor( x ) + T(1.0) + w3(x)) / (w2(x) + w3(x))) - x;
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_COMMON_H
