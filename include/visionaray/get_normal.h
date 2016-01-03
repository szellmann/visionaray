// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_NORMAL
#define VSNRAY_GET_NORMAL 1

#include <iterator>

#include "math/math.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Get face normal from array
//

template <typename Normals, typename HR, typename Primitive>
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        Primitive                   /* */,
        normals_per_face_binding    /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    return normals[hr.prim_id];
}


//-------------------------------------------------------------------------------------------------
// Get triangle vertex normal from array
//

template <typename Normals, typename HR, typename T>
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        basic_triangle<3, T>        /* */,
        normals_per_vertex_binding  /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    return normalize( lerp(
            normals[hr.prim_id * 3],
            normals[hr.prim_id * 3 + 1],
            normals[hr.prim_id * 3 + 2],
            hr.u,
            hr.v
            ) );
}


//-------------------------------------------------------------------------------------------------
// Gather four face normals with SSE
//

template <
    typename Normals,
    template <typename, typename> class HR,
    typename HRP,
    typename Primitive
    >
inline vector<3, simd::float4> get_normal(
        Normals                     normals,
        HR<simd::ray4, HRP> const&  hr,
        Primitive                   /* */,
        normals_per_face_binding    /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr4 = unpack(hr);

    auto get_norm = [&](int x)
    {
        return hr4[x].hit ? normals[hr4[x].prim_id] : N();
    };

    auto n1 = get_norm(0);
    auto n2 = get_norm(1);
    auto n3 = get_norm(2);
    auto n4 = get_norm(3);

    return vector<3, simd::float4>(
            simd::float4( n1.x, n2.x, n3.x, n4.x ),
            simd::float4( n1.y, n2.y, n3.y, n4.y ),
            simd::float4( n1.z, n2.z, n3.z, n4.z )
            );
}

//-------------------------------------------------------------------------------------------------
// Gather four triangle vertex normals with SSE
//

template <typename Normals, typename T>
inline vector<3, simd::float4> get_normal(
        Normals                                             normals,
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, T>                                /* */,
        normals_per_vertex_binding                          /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr4 = unpack(hr);

    auto get_norm = [&](int x, int y)
    {
        return hr4[x].hit ? normals[hr4[x].prim_id * 3 + y] : N();
    };

    vector<3, simd::float4> n1(
            simd::float4( get_norm(0, 0).x, get_norm(1, 0).x, get_norm(2, 0).x, get_norm(3, 0).x ),
            simd::float4( get_norm(0, 0).y, get_norm(1, 0).y, get_norm(2, 0).y, get_norm(3, 0).y ),
            simd::float4( get_norm(0, 0).z, get_norm(1, 0).z, get_norm(2, 0).z, get_norm(3, 0).z )
            );

    vector<3, simd::float4> n2(
            simd::float4( get_norm(0, 1).x, get_norm(1, 1).x, get_norm(2, 1).x, get_norm(3, 1).x ),
            simd::float4( get_norm(0, 1).y, get_norm(1, 1).y, get_norm(2, 1).y, get_norm(3, 1).y ),
            simd::float4( get_norm(0, 1).z, get_norm(1, 1).z, get_norm(2, 1).z, get_norm(3, 1).z )
            );

    vector<3, simd::float4> n3(
            simd::float4( get_norm(0, 2).x, get_norm(1, 2).x, get_norm(2, 2).x, get_norm(3, 2).x ),
            simd::float4( get_norm(0, 2).y, get_norm(1, 2).y, get_norm(2, 2).y, get_norm(3, 2).y ),
            simd::float4( get_norm(0, 2).z, get_norm(1, 2).z, get_norm(2, 2).z, get_norm(3, 2).z )
            );

    return normalize( lerp(n1, n2, n3, hr.u, hr.v) );
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather eight face normals with AVX
//

template <
    typename Normals,
    template <typename, typename> class HR,
    typename HRP,
    typename Primitive
    >
inline vector<3, simd::float8> get_normal(
        Normals                     normals,
        HR<simd::ray8, HRP> const&  hr,
        Primitive                   /* */,
        normals_per_face_binding    /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr8 = unpack(hr);

    auto get_norm = [&](int x)
    {
        return hr8[x].hit ? normals[hr8[x].prim_id] : N();
    };

    auto n1 = get_norm(0);
    auto n2 = get_norm(1);
    auto n3 = get_norm(2);
    auto n4 = get_norm(3);
    auto n5 = get_norm(4);
    auto n6 = get_norm(5);
    auto n7 = get_norm(6);
    auto n8 = get_norm(7);

    return vector<3, simd::float8>(
            simd::float8( n1.x, n2.x, n3.x, n4.x, n5.x, n6.x, n7.x, n8.x ),
            simd::float8( n1.y, n2.y, n3.y, n4.y, n5.y, n6.y, n7.y, n8.y ),
            simd::float8( n1.z, n2.z, n3.z, n4.z, n5.z, n6.z, n7.z, n8.z )
            );
}

//-------------------------------------------------------------------------------------------------
// Gather eight triangle vertex normals with AVX
//

template <typename Normals, typename T>
inline vector<3, simd::float8> get_normal(
        Normals                                             normals,
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        basic_triangle<3, T>                                /* */,
        normals_per_vertex_binding                          /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr8 = unpack(hr);

    auto get_norm = [&](int x, int y)
    {
        return hr8[x].hit ? normals[hr8[x].prim_id * 3 + y] : N();
    };

    vector<3, simd::float8> n1(
            simd::float8( get_norm(0, 0).x, get_norm(1, 0).x, get_norm(2, 0).x, get_norm(3, 0).x,
                          get_norm(4, 0).x, get_norm(5, 0).x, get_norm(6, 0).x, get_norm(7, 0).x ),
            simd::float8( get_norm(0, 0).y, get_norm(1, 0).y, get_norm(2, 0).y, get_norm(3, 0).y,
                          get_norm(4, 0).y, get_norm(5, 0).y, get_norm(6, 0).y, get_norm(7, 0).y ),
            simd::float8( get_norm(0, 0).z, get_norm(1, 0).z, get_norm(2, 0).z, get_norm(3, 0).z,
                          get_norm(4, 0).z, get_norm(5, 0).z, get_norm(6, 0).z, get_norm(7, 0).z )
            );

    vector<3, simd::float8> n2(
            simd::float8( get_norm(0, 1).x, get_norm(1, 1).x, get_norm(2, 1).x, get_norm(3, 1).x,
                          get_norm(4, 1).x, get_norm(5, 1).x, get_norm(6, 1).x, get_norm(7, 1).x ),
            simd::float8( get_norm(0, 1).y, get_norm(1, 1).y, get_norm(2, 1).y, get_norm(3, 1).y,
                          get_norm(4, 1).y, get_norm(5, 1).y, get_norm(6, 1).y, get_norm(7, 1).y ),
            simd::float8( get_norm(0, 1).z, get_norm(1, 1).z, get_norm(2, 1).z, get_norm(3, 1).z,
                          get_norm(4, 1).z, get_norm(5, 1).z, get_norm(6, 1).z, get_norm(7, 1).z )
            );

    vector<3, simd::float8> n3(
            simd::float8( get_norm(0, 2).x, get_norm(1, 2).x, get_norm(2, 2).x, get_norm(3, 2).x,
                          get_norm(4, 2).x, get_norm(5, 2).x, get_norm(6, 2).x, get_norm(7, 2).x ),
            simd::float8( get_norm(0, 2).y, get_norm(1, 2).y, get_norm(2, 2).y, get_norm(3, 2).y,
                          get_norm(4, 2).y, get_norm(5, 2).y, get_norm(6, 2).y, get_norm(7, 2).y ),
            simd::float8( get_norm(0, 2).z, get_norm(1, 2).z, get_norm(2, 2).z, get_norm(3, 2).z,
                          get_norm(4, 2).z, get_norm(5, 2).z, get_norm(6, 2).z, get_norm(7, 2).z )
            );

    return normalize( lerp(n1, n2, n3, hr.u, hr.v) );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Get normal from plane primitive
//

template <typename HR, typename T, typename NormalBinding>
VSNRAY_FUNC
inline vector<3, T> get_normal(
        HR const&                   hr,
        basic_plane<3, T> const&    plane,
        NormalBinding               /* */
        )
{
    VSNRAY_UNUSED(hr);

    return plane.normal;
}


//-------------------------------------------------------------------------------------------------
// Get normal on sphere surface
//

template <typename HR, typename T, typename NormalBinding>
VSNRAY_FUNC
inline vector<3, T> get_normal(
        HR const&               hr,
        basic_sphere<T> const&  sphere,
        NormalBinding           /* */
        )
{
    return (hr.isect_pos - sphere.center) / sphere.radius;
}


//-------------------------------------------------------------------------------------------------
// w/o tag dispatch default to triangles
//

template <typename Normals, typename HR, typename NormalBinding>
VSNRAY_FUNC
inline auto get_normal(Normals normals, HR const& hr, NormalBinding /* */)
    -> decltype( get_normal(
            normals,
            hr,
            basic_triangle<3, typename HR::value_type>{},
            NormalBinding{}
            ) )
{
    return get_normal(
            normals,
            hr,
            basic_triangle<3, typename HR::value_type>{},
            NormalBinding{}
            );
}

} // visionaray

#endif // VSNRAY_GET_NORMAL
