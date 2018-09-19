// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_COLOR_H
#define VSNRAY_GET_COLOR_H 1

#include <iterator>
#include <type_traits>

#include "math/simd/type_traits.h"
#include "math/array.h"
#include "math/triangle.h"
#include "math/vector.h"
#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Get face color from array
//

template <
    typename Colors,
    typename HR,
    typename T = typename HR::scalar_type,
    typename Primitive,
    typename = typename std::enable_if<!simd::is_simd_vector<T>::value>::type
    >
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
    typename HR,
    typename T = typename HR::scalar_type,
    typename Primitive,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline vector<3, T> get_color(
        Colors                  colors,
        HR const&               hr,
        Primitive               /* */,
        colors_per_face_binding /* */
        )
{
    using C = typename std::iterator_traits<Colors>::value_type;
    using float_array = simd::aligned_array_t<T>;

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
// Triangle and SIMD ray, colors per vertex
//

template <
    typename Colors,
    typename HR,
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
inline auto get_color(
        Colors                    colors,
        HR const&                 hr,
        basic_triangle<3, T>      /* */,
        colors_per_vertex_binding /* */
        )
    -> vector<3, typename HR::scalar_type>
{
    using U = typename HR::scalar_type;
    using C = typename std::iterator_traits<Colors>::value_type;
    using float_array = simd::aligned_array_t<U>;

    auto hrs = unpack(hr);

    auto get_clr = [&](int x, int y)
    {
        return hrs[x].hit ? colors[hrs[x].prim_id * 3 + y] : C();
    };

    float_array x1;
    float_array y1;
    float_array z1;

    float_array x2;
    float_array y2;
    float_array z2;

    float_array x3;
    float_array y3;
    float_array z3;

    for (size_t i = 0; i < simd::num_elements<U>::value; ++i)
    {
        auto cc1 = get_clr(i, 0);
        auto cc2 = get_clr(i, 1);
        auto cc3 = get_clr(i, 2);

        x1[i] = cc1.x;
        y1[i] = cc1.y;
        z1[i] = cc1.z;

        x2[i] = cc2.x;
        y2[i] = cc2.y;
        z2[i] = cc2.z;

        x3[i] = cc3.x;
        y3[i] = cc3.y;
        z3[i] = cc3.z;
    }

    vector<3, U> c1(x1, y1, z1);
    vector<3, U> c2(x2, y2, z2);
    vector<3, U> c3(x3, y3, z3);

    return lerp( c1, c2, c3, hr.u, hr.v );
}


//-------------------------------------------------------------------------------------------------
// Gather N vertex colors from array
//

template <
    typename Colors,
    typename HR,
    size_t N,
    typename Primitive,
    typename ColorBinding
    >
VSNRAY_FUNC
inline auto get_color(
        Colors              colors,
        array<HR, N> const& hr,
        Primitive           /* */,
        ColorBinding        /* */
        )
    -> array<typename std::iterator_traits<Colors>::value_type, N>
{
    using C = typename std::iterator_traits<Colors>::value_type;

    array<C, N> result;

    for (size_t i = 0; i < N; ++i)
    {
        result[i] = get_color(colors, hr[i], Primitive{}, ColorBinding{});
    }

    return result;
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
            basic_triangle<3, typename HR::scalar_type>{},
            ColorBinding{}
            ) )
{
    return get_color(
            colors,
            hr,
            basic_triangle<3, typename HR::scalar_type>{},
            ColorBinding{}
            );
}

} // visionaray

#endif // VSNRAY_GET_COLOR_H
