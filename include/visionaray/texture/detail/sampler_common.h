// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SAMPLER_COMMON_H
#define VSNRAY_SAMPLER_COMMON_H 1


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
F map_tex_coord(F const& coord, I const& texsize, tex_address_mode mode)
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
F map_tex_coord(F const& coord, I const& texsize, std::array<tex_address_mode, 1> const& mode)
{
    return map_tex_coord(coord, texsize, mode[0]);
}

template <size_t Dim, typename F, typename I>
vector<Dim, F> map_tex_coord(
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
    return tex[idx];
}


// SIMD: if multi-channel texture, assume AoS

template <typename T>
inline simd::float4 point(T const* tex, simd::int4 const& idx, simd::float4 /* result type */)
{
    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], idx);
    return simd::float4(
            tex[indices[0]],
            tex[indices[1]],
            tex[indices[2]],
            tex[indices[3]]
            );
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename T>
inline simd::float8 point(T const* tex, simd::int8 const& idx, simd::float8 /* result type */)
{
    VSNRAY_ALIGN(32) int indices[8];
    store(&indices[0], idx);
    return simd::float8(
            tex[indices[0]],
            tex[indices[1]],
            tex[indices[2]],
            tex[indices[3]],
            tex[indices[4]],
            tex[indices[5]],
            tex[indices[6]],
            tex[indices[7]]
            );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


inline vector<3, simd::float4> point(
        vector<3, unorm<8>> const*  tex,
        simd::int4 const&           idx,
        vector<3, simd::float4>     /* result type */
        )
{
    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], idx);
    return vector<3, simd::float4>(
            simd::float4(tex[indices[0]].x, tex[indices[1]].x, tex[indices[2]].x, tex[indices[3]].x),
            simd::float4(tex[indices[0]].y, tex[indices[1]].y, tex[indices[2]].y, tex[indices[3]].y),
            simd::float4(tex[indices[0]].z, tex[indices[1]].z, tex[indices[2]].z, tex[indices[3]].z)
            );
}

inline vector<4, simd::float4> point(
        vector<4, float> const*     tex,
        simd::int4 const&           idx,
        vector<4, simd::float4>     /* result type */
        )
{

    // Special case: colors are AoS. Those can be obtained
    // without a context switch to GP registers by transposing
    // to SoA after memory lookup.

    simd::float4 iidx( idx * 4 );
    VSNRAY_ALIGN(16) float indices[4];
    store(&indices[0], iidx);

    float const* tmp = reinterpret_cast<float const*>(tex);

    vector<4, simd::float4> colors(
            &tmp[0] + static_cast<int>(indices[0]),
            &tmp[0] + static_cast<int>(indices[1]),
            &tmp[0] + static_cast<int>(indices[2]),
            &tmp[0] + static_cast<int>(indices[3])
            );

    colors = transpose(colors);
    return colors;

}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

inline vector<4, simd::float8> point(
        vector<4, float> const*     tex,
        simd::int8 const&           idx,
        vector<4, simd::float8>     /* result type */
        )
{
    VSNRAY_ALIGN(32) int indices[8];
    store(&indices[0], idx);

    std::array<vector<4, float>, 8> arr{{
            tex[indices[0]],
            tex[indices[1]],
            tex[indices[2]],
            tex[indices[3]],
            tex[indices[4]],
            tex[indices[5]],
            tex[indices[6]],
            tex[indices[7]]
            }};


    return simd::pack(arr);
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


// SIMD: special case, if multi-channel texture, assume SoA

inline simd::float4 point(
        simd::float4 const* tex,
        int                 coord,
        simd::float4        /* result type */
        )
{
    return tex[coord];
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2

//-------------------------------------------------------------------------------------------------
// Point sampling using AVX2 gather instructions
//

inline simd::float4 point(float const* tex, simd::int4 const& idx, simd::float4 /* result type */)
{
    return _mm_i32gather_ps(tex, idx, 4);
}

inline simd::float8 point(float const* tex, simd::int8 const& idx, simd::float8 /* result type */)
{
    return _mm256_i32gather_ps(tex, idx, 4);
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2



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
