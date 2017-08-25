// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_TEX_COORD_H
#define VSNRAY_GET_TEX_COORD_H 1

#include <array>
#include <iterator>
#include <type_traits>

#include "math/detail/math.h"
#include "math/simd/type_traits.h"
#include "math/intersect.h"
#include "math/primitive.h"
#include "math/triangle.h"
#include "math/vector.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Triangle
//

template <typename TexCoords, typename R, typename T>
VSNRAY_FUNC
inline auto get_tex_coord(
        TexCoords                                   tex_coords,
        hit_record<R, primitive<unsigned>> const&   hr,
        basic_triangle<3, T>                        /* */
        )
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
    typename T,
    typename U,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline vector<2, T> get_tex_coord(
        TexCoords                                               coords,
        hit_record<basic_ray<T>, primitive<unsigned>> const&    hr,
        basic_triangle<3, U>                                    /* */
        )
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;
    using float_array = simd::aligned_array_t<T>;

    auto hrs = unpack(hr);

    auto get_coord = [&](int x, int y)
    {
        return hrs[x].hit ? coords[hrs[x].prim_id * 3 + y] : TC();
    };


    float_array x1;
    float_array y1;
    float_array x2;
    float_array y2;
    float_array x3;
    float_array y3;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        x1[i] = get_coord(i, 0).x;
        y1[i] = get_coord(i, 0).y;

        x2[i] = get_coord(i, 1).x;
        y2[i] = get_coord(i, 1).y;

        x3[i] = get_coord(i, 2).x;
        y3[i] = get_coord(i, 2).y;
    }

    vector<2, T> tc1(x1, y1);
    vector<2, T> tc2(x2, y2);
    vector<2, T> tc3(x3, y3);

    return lerp( tc1, tc2, tc3, hr.u, hr.v );
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
inline auto get_tex_coord(
        TexCoords                   coords,
        std::array<HR, N> const&    hr,
        Primitive                   /* */
        )
    -> std::array<typename std::iterator_traits<TexCoords>::value_type, N>
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;

    std::array<TC, N> result;

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
