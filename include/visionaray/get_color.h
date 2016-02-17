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
// Gather N face colors for SIMD ray
//

template <
    typename Colors,
    template <typename, typename> class HR,
    typename T,
    typename HRP,
    typename Primitive,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline vector<3, simd::float4> get_color(
        Colors                          colors,
        HR<basic_ray<T>, HRP> const&    hr,
        Primitive                       /* */,
        colors_per_face_binding         /* */
        )
{
    using C = typename std::iterator_traits<Colors>::value_type;
    using float_array = typename simd::aligned_array<T>::type;

    auto hrs = unpack(hr);

    float_array x;
    float_array y;
    float_array z;

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto c = hrs[i].hit ? colors[hrs[i].prim_id] : C();
        x[i] = c.x;
        y[i] = c.y;
        z[i] = c.z;
    }

    return vector<3, T>(
            T(x),
            T(y),
            T(z)
            );
}


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
