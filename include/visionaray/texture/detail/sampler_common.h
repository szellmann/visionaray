// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SAMPLER_COMMON_H
#define VSNRAY_SAMPLER_COMMON_H 1

#include <array>
#include <type_traits>

#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>

#include "../forward.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Remap tex coord based on address mode
//

template <typename F, typename I>
inline F map_tex_coord(F const& coord, I const& texsize, tex_address_mode mode)
{
    F N = convert_to_float(texsize);

    switch (mode)
    {

    case Mirror:
        return select(
            (convert_to_int(floor(coord)) & I(1)) == 1, // if is odd
            F(texsize - 1) / F(texsize) - (coord - floor(coord)),
            coord - floor(coord)
            );

    case Wrap:
        return coord - floor(coord);

    case Clamp:
        // fall-through
    default:
        return clamp( coord, F(0.0), F(1.0) - F(1.0) / N );
    }
}

template <typename F, typename I>
inline F map_tex_coord(F const& coord, I const& texsize, std::array<tex_address_mode, 1> const& mode)
{
    return map_tex_coord(coord, texsize, mode[0]);
}

template <size_t Dim, typename F, typename I>
inline vector<Dim, F> map_tex_coord(
        vector<Dim, F> const&                       coord,
        vector<Dim, I> const&                       texsize,
        std::array<tex_address_mode, Dim> const&    mode
        )
{
    vector<Dim, F> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = map_tex_coord( coord[d], texsize[d], mode[d] );
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Array access functions for scalar and SIMD types
// Return type as 3rd value for overload resolution!
//

template <typename RT, typename T>
inline RT point(T const* tex, ptrdiff_t idx, RT = RT())
{
    return RT(tex[idx]);
}


// SIMD: if multi-channel texture, assume AoS

template <
    typename T,
    typename IndexT,
    typename RT,
    typename = typename std::enable_if<simd::is_simd_vector<IndexT>::value>::type
    >
inline RT point(
        T const*        tex,
        IndexT const&   index,
        RT              /* */
        )
{
    return simd::gather(tex, index);
}


// SIMD: special case, if multi-channel texture, assume SoA

inline simd::float4 point(
        simd::float4 const* tex,
        int                 coord,
        simd::float4        /* result type */
        )
{
    return tex[coord];
}


//-------------------------------------------------------------------------------------------------
// Weight functions for higher order texture interpolation
//

namespace bspline
{

// weight functions for Mitchell - Netravalli B-Spline with B = 1 and C = 0

template <typename T>
struct w0_func
{
    inline T operator()(T const& a)
    {
        return T( (1.0 / 6.0) * (-(a * a * a) + 3.0 * a * a - 3.0 * a + 1.0) );
    }
};

template <typename T>
struct w1_func
{
    inline T operator()(T const& a)
    {
        return T( (1.0 / 6.0) * (3.0 * a * a * a - 6.0 * a * a + 4.0) );
    }
};

template <typename T>
struct w2_func
{
    inline T operator()(T const& a)
    {
        return T( (1.0 / 6.0) * (-3.0 * a * a * a + 3.0 * a * a + 3.0 * a + 1.0) );
    }
};

template <typename T>
struct w3_func
{
    inline T operator()(T const& a)
    {
        return T( (1.0 / 6.0) * (a * a * a) );
    }
};

} // bspline

namespace cspline
{

// weight functions for Catmull - Rom Cardinal Spline

template <typename T>
struct w0_func
{
    inline T operator()(T const& a)
    {
        return T( -0.5 * a * a * a + a * a - 0.5 * a );
    }
};

template <typename T>
struct w1_func
{
    inline T operator()(T const& a)
    {
        return T( 1.5 * a * a * a - 2.5 * a * a + 1.0 );
    }
};

template <typename T>
struct w2_func
{
    inline T operator()(T const& a)
    {
        return T( -1.5 * a * a * a + 2.0 * a * a + 0.5 * a );
    }
};

template <typename T>
struct w3_func
{
    inline T operator()(T const& a)
    {
        return T( 0.5 * a * a * a - 0.5 * a * a );
    }
};

} // cspline

// helper functions for cubic interpolation
template <typename T>
inline T g0(T const& x)
{
    bspline::w0_func<T> w0;
    bspline::w1_func<T> w1;
    return w0(x) + w1(x);
}

template <typename T>
inline T g1(T const& x)
{
    bspline::w2_func<T> w2;
    bspline::w3_func<T> w3;
    return w2(x) + w3(x);
}

template <typename T>
inline T h0(T const& x)
{
    bspline::w0_func<T> w0;
    bspline::w1_func<T> w1;
    return ((floor( x ) - T(1.0) + w1(x)) / (w0(x) + w1(x))) + x;
}

template <typename T>
inline T h1(T const& x)
{
    bspline::w2_func<T> w2;
    bspline::w3_func<T> w3;
    return ((floor( x ) + T(1.0) + w3(x)) / (w2(x) + w3(x))) - x;
}

} // detail
} // visionaray

#endif // VSNRAY_SAMPLER_COMMON_H
