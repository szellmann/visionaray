// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_COLOR_H
#define VSNRAY_GET_COLOR_H 1

#include <array>
#include <iterator>

#include "math/math.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Get face color from array
//

template <typename Colors, typename HR, typename Primitive>
VSNRAY_FUNC
inline auto get_color(
        Colors                  colors,
        HR const&               hr,
        Primitive               /* */,
        colors_per_face_binding /* */
        )
    -> typename std::iterator_traits<Colors>::value_type
{
    return colors[hr.prim_id];
}


//-------------------------------------------------------------------------------------------------
// Get triangle vertex color from array
//

template <typename Colors, typename HR, typename T>
VSNRAY_FUNC
inline auto get_color(
        Colors                      colors,
        HR const&                   hr,
        basic_triangle<3, T>        /* */,
        colors_per_vertex_binding   /* */
        )
    -> typename std::iterator_traits<Colors>::value_type
{
    return lerp(
            colors[hr.prim_id * 3],
            colors[hr.prim_id * 3 + 1],
            colors[hr.prim_id * 3 + 2],
            hr.u,
            hr.v
            );
}


//-------------------------------------------------------------------------------------------------
// Gather four face colors with SSE
//

template <
    typename Colors,
    template <typename, typename> class HR,
    typename HRP,
    typename Primitive
    >
inline vector<3, simd::float4> get_color(
        Colors                      colors,
        HR<simd::ray4, HRP> const&  hr,
        Primitive                   /* */,
        normals_per_face_binding    /* */
        )
{
    using C = typename std::iterator_traits<Colors>::value_type;

    auto hr4 = unpack(hr);

    auto get_clr = [&](int x)
    {
        return hr4[x].hit ? colors[hr4[x].prim_id] : C();
    };

    auto c1 = get_clr(0);
    auto c2 = get_clr(1);
    auto c3 = get_clr(2);
    auto c4 = get_clr(3);

    return vector<3, simd::float4>(
            simd::float4( c1.x, c2.x, c3.x, c4.x ),
            simd::float4( c1.y, c2.y, c3.y, c4.y ),
            simd::float4( c1.z, c2.z, c3.z. c4.z )
            );
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather eight face colors with AVX
//

template <
    typename Colors,
    template <typename, typename> class HR,
    typename HRP,
    typename Primitive
    >
inline vector<3, simd::float8> get_color(
        Colors                      colors,
        HR<simd::ray8, HRP> const&  hr,
        Primitive                   /* */,
        normals_per_face_binding    /* */
        )
{
    using C = typename std::iterator_traits<Colors>::value_type;

    auto hr8 = unpack(hr);

    auto get_clr = [&](int x)
    {
        return hr8[x].hit ? colors[hr8[x].prim_id] : C();
    };

    auto c1 = get_clr(0);
    auto c2 = get_clr(1);
    auto c3 = get_clr(2);
    auto c4 = get_clr(3);
    auto c5 = get_clr(4);
    auto c6 = get_clr(5);
    auto c7 = get_clr(6);
    auto c8 = get_clr(7);

    return vector<3, simd::float8>(
            simd::float8( c1.x, c2.x, c3.x, c4.x, c5.x, c6.x, c7.x, c8.x ),
            simd::float8( c1.y, c2.y, c3.y, c4.y, c5.y, c6.y, c7.y, c8.y ),
            simd::float8( c1.z, c2.z, c3.z. c4.z, c5.z, c6.z, c7.z, c8.z )
            );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// SSE triangle, colors per vertex
//

template <typename Colors, typename T>
inline vector<3, simd::float4> get_color(
        Colors                                              colors,
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, T>                                /* */,
        colors_per_vertex_binding                           /* */
        )
{
    using C = typename std::iterator_traits<Colors>::value_type;

    auto hr4 = unpack(hr);

    auto get_clr = [&](int x, int y)
    {
        return hr4[x].hit ? colors[hr4[x].prim_id * 3 + y] : C();
    };

    vector<3, simd::float4> c1(
            simd::float4( get_clr(0, 0).x, get_clr(1, 0).x, get_clr(2, 0).x, get_clr(3, 0).x ),
            simd::float4( get_clr(0, 0).y, get_clr(1, 0).y, get_clr(2, 0).y, get_clr(3, 0).y ),
            simd::float4( get_clr(0, 0).z, get_clr(1, 0).z, get_clr(2, 0).y, get_clr(3, 0).z )
            );

    vector<3, simd::float4> c2(
            simd::float4( get_clr(0, 1).x, get_clr(1, 1).x, get_clr(2, 1).x, get_clr(3, 1).x ),
            simd::float4( get_clr(0, 1).y, get_clr(1, 1).y, get_clr(2, 1).y, get_clr(3, 1).y ),
            simd::float4( get_clr(0, 1).z, get_clr(1, 1).z, get_clr(2, 1).z, get_clr(3, 1).z )
            );

    vector<3, simd::float4> c3(
            simd::float4( get_clr(0, 2).x, get_clr(1, 2).x, get_clr(2, 2).x, get_clr(3, 2).x ),
            simd::float4( get_clr(0, 2).y, get_clr(1, 2).y, get_clr(2, 2).y, get_clr(3, 2).y ),
            simd::float4( get_clr(0, 2).z, get_clr(1, 2).z, get_clr(2, 2).z, get_clr(3, 2).z )
            );

    return lerp( c1, c2, c3, hr.u, hr.v );
}


//-------------------------------------------------------------------------------------------------
// Gather four vertex colors from array
//

template <
    typename Colors,
    typename HR
    >
inline auto get_color(
        Colors                      colors,
        std::array<HR, 4> const&    hr,
        basic_triangle<3, float>    /* */,
        colors_per_vertex_binding   /* */
        )
    -> std::array<typename std::iterator_traits<Colors>::value_type, 4>
{
    using C = typename std::iterator_traits<Colors>::value_type;
    using ColorBinding = colors_per_vertex_binding;
    using Primitive = basic_triangle<3, float>; // TODO: make this work for other planar surfaces!

    return std::array<C, 4>{{
            get_color(colors, hr[0], Primitive{}, ColorBinding{}),
            get_color(colors, hr[1], Primitive{}, ColorBinding{}),
            get_color(colors, hr[2], Primitive{}, ColorBinding{}),
            get_color(colors, hr[3], Primitive{}, ColorBinding{})
            }};
}

//-------------------------------------------------------------------------------------------------
// w/o tag dispatch default to triangles
//

template <typename Colors, typename HR, typename ColorBinding>
VSNRAY_FUNC
inline auto get_color(Colors colors, HR const& hr, ColorBinding /* */)
    -> decltype( get_color(
            colors,
            hr,
            basic_triangle<3, typename HR::value_type>{},
            ColorBinding{}
            ) )
{
    return get_color(
            colors,
            hr,
            basic_triangle<3, typename HR::value_type>{},
            ColorBinding{}
            );
}

} // visionaray

#endif // VSNRAY_GET_COLOR_H
