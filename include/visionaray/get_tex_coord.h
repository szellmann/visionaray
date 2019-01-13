// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_TEX_COORD_H
#define VSNRAY_GET_TEX_COORD_H 1

#include <iterator>
#include <type_traits>

#include "detail/macros.h"
#include "math/detail/math.h"
#include "math/simd/type_traits.h"
#include "math/array.h"
#include "math/constants.h"
#include "math/sphere.h"
#include "math/triangle.h"
#include "math/vector.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Default get_tex_coord implementation, assumes that tex coord list is unused!
//

template <typename TexCoords, typename HR, typename Primitive>
VSNRAY_FUNC
inline auto get_tex_coord(TexCoords tex_coords, HR const& hr, Primitive const& prim)
    -> typename std::iterator_traits<TexCoords>::value_type
{
    VSNRAY_UNUSED(tex_coords);

    return get_tex_coord(hr, prim);
}


//-------------------------------------------------------------------------------------------------
// Triangle
//

template <
    typename TexCoords,
    typename HR,
    typename T,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_tex_coord(TexCoords tex_coords, HR const& hr, basic_triangle<3, T> /* */)
    -> typename std::iterator_traits<TexCoords>::value_type
{
    return lerp(
            tex_coords[hr.prim_id * 3],
            tex_coords[hr.prim_id * 3 + 1],
            tex_coords[hr.prim_id * 3 + 2],
            hr.u,
            hr.v
            );
}


//-------------------------------------------------------------------------------------------------
// SIMD triangle
//

template <
    typename TexCoords,
    typename HR,
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_tex_coord(TexCoords tex_coords, HR const& hr, basic_triangle<3, T> /* */)
    -> vector<2, typename HR::scalar_type>
{
    using U = typename HR::scalar_type;
    using TC = typename std::iterator_traits<TexCoords>::value_type;
    using float_array = simd::aligned_array_t<U>;

    auto hrs = unpack(hr);

    auto get_coord = [&](int x, int y)
    {
        return hrs[x].hit ? tex_coords[hrs[x].prim_id * 3 + y] : TC();
    };


    float_array x1;
    float_array y1;
    float_array x2;
    float_array y2;
    float_array x3;
    float_array y3;

    for (int i = 0; i < simd::num_elements<U>::value; ++i)
    {
        x1[i] = get_coord(i, 0).x;
        y1[i] = get_coord(i, 0).y;

        x2[i] = get_coord(i, 1).x;
        y2[i] = get_coord(i, 1).y;

        x3[i] = get_coord(i, 2).x;
        y3[i] = get_coord(i, 2).y;
    }

    vector<2, U> tc1(x1, y1);
    vector<2, U> tc2(x2, y2);
    vector<2, U> tc3(x3, y3);

    return lerp( tc1, tc2, tc3, hr.u, hr.v );
}


//-------------------------------------------------------------------------------------------------
// Sphere
//

template <
    typename HR,
    typename T,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_tex_coord(HR const& hr, basic_sphere<T> const& sphere)
    -> vector<2, typename HR::scalar_type>
{
    using S = typename HR::scalar_type;

    auto n = (hr.isect_pos - sphere.center) / sphere.radius;

    return vector<2, S>(
            atan2(n.x, n.z) / constants::two_pi<S>() + S(0.5),
            n.y * S(0.5) + S(0.5)
            );
}


//-------------------------------------------------------------------------------------------------
// Gather N texture coordinates from array
//

template <
    typename TexCoords,
    typename HR,
    size_t N,
    typename Primitive
    >
VSNRAY_FUNC
inline auto get_tex_coord(
        TexCoords           coords,
        array<HR, N> const& hr,
        Primitive           /* */
        )
    -> array<typename std::iterator_traits<TexCoords>::value_type, N>
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;

    array<TC, N> result;

    for (size_t i = 0; i < N; ++i)
    {
        result[i] = get_tex_coord(coords, hr[i], Primitive{});
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// w/o tag dispatch default to triangles
//

template <typename TexCoords, typename HR>
VSNRAY_FUNC
inline auto get_tex_coord(TexCoords coords, HR const& hr)
    -> decltype(get_tex_coord(coords, hr, basic_triangle<3, typename HR::scalar_type>{}))
{
    return get_tex_coord(coords, hr, basic_triangle<3, typename HR::scalar_type>{});
}

} // visionaray

#endif // VSNRAY_GET_TEX_COORD_H
