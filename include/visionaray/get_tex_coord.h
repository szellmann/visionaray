// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_TEX_COORD_H
#define VSNRAY_GET_TEX_COORD_H 1

#include <array>
#include <iterator>

#include <visionaray/math/math.h>


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
// SSE triangle
//

template <typename TexCoords, typename T>
inline vector<2, simd::float4> get_tex_coord(
        TexCoords                                           coords,
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, T>                                /* */
        )
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;

    auto hr4 = simd::unpack(hr);

    auto get_coord = [&](int x, int y)
    {
        return hr4[x].hit ? coords[hr4[x].prim_id * 3 + y] : TC();
    };

    vector<2, simd::float4> tc1(
            simd::float4( get_coord(0, 0).x, get_coord(1, 0).x, get_coord(2, 0).x, get_coord(3, 0).x ),
            simd::float4( get_coord(0, 0).y, get_coord(1, 0).y, get_coord(2, 0).y, get_coord(3, 0).y )
            );

    vector<2, simd::float4> tc2(
            simd::float4( get_coord(0, 1).x, get_coord(1, 1).x, get_coord(2, 1).x, get_coord(3, 1).x ),
            simd::float4( get_coord(0, 1).y, get_coord(1, 1).y, get_coord(2, 1).y, get_coord(3, 1).y )
            );

    vector<2, simd::float4> tc3(
            simd::float4( get_coord(0, 2).x, get_coord(1, 2).x, get_coord(2, 2).x, get_coord(3, 2).x ),
            simd::float4( get_coord(0, 2).y, get_coord(1, 2).y, get_coord(2, 2).y, get_coord(3, 2).y )
            );

    return lerp( tc1, tc2, tc3, hr.u, hr.v );
}


//-------------------------------------------------------------------------------------------------
// Gather four texture coordinates from array
//

template <
    typename TexCoords,
    typename HR,
    typename Primitive
    >
inline auto get_tex_coord(
        TexCoords                   coords,
        std::array<HR, 4> const&    hr,
        Primitive                   /* */
        )
    -> std::array<typename std::iterator_traits<TexCoords>::value_type, 4>
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;

    return std::array<TC, 4>{{
            get_tex_coord(coords, hr[0], Primitive{}),
            get_tex_coord(coords, hr[1], Primitive{}),
            get_tex_coord(coords, hr[2], Primitive{}),
            get_tex_coord(coords, hr[3], Primitive{})
            }};
}


//-------------------------------------------------------------------------------------------------
// w/o tag dispatch default to triangles
//

template <typename TexCoords, typename HR>
inline auto get_tex_coord(TexCoords coords, HR const& hr)
    -> decltype(get_tex_coord(coords, hr, basic_triangle<3, typename HR::value_type>{}))
{
    return get_tex_coord(coords, hr, basic_triangle<3, typename HR::value_type>{});
}

} // visionaray

#endif // VSNRAY_GET_TEX_COORD_H
