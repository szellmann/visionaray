// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_NORMAL
#define VSNRAY_GET_NORMAL 1

#include <iterator>
#include <type_traits>

#include "math/math.h"
#include "prim_traits.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Get face normal from array
//

template <
    typename Normals,
    typename HR,
    typename Primitive,
    typename = typename std::enable_if<num_normals<Primitive, per_face_binding>::value == 1>::type
    >
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        Primitive                   /* */,
        per_face_binding            /* */
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
        per_vertex_binding          /* */
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
// Gather N face normals for SIMD ray
//

template <
    typename Normals,
    template <typename, typename> class HR,
    typename T,
    typename HRP,
    typename Primitive,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline vector<3, T> get_normal(
        Normals                         normals,
        HR<basic_ray<T>, HRP> const&    hr,
        Primitive                       /* */,
        per_face_binding                /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using float_array = typename simd::aligned_array<T>::type;

    auto hrs = unpack(hr);

    float_array x;
    float_array y;
    float_array z;

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto n = hrs[i].hit ? normals[hrs[i].prim_id] : N();
        x[i] = n.x;
        y[i] = n.y;
        z[i] = n.z;
    }

    return vector<3, T>(
            T(x),
            T(y),
            T(z)
            );
}

//-------------------------------------------------------------------------------------------------
// Gather N triangle vertex normals for SIMD ray
//

template <
    typename Normals,
    typename T,
    typename U,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline vector<3, T> get_normal(
        Normals                                                 normals,
        hit_record<basic_ray<T>, primitive<unsigned>> const&    hr,
        basic_triangle<3, U>                                    /* */,
        per_vertex_binding                                      /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using float_array = typename simd::aligned_array<T>::type;

    auto hrs = unpack(hr);

    auto get_norm = [&](int x, int y)
    {
        return hrs[x].hit ? normals[hrs[x].prim_id * 3 + y] : N();
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

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto nn1 = get_norm(i, 0);
        auto nn2 = get_norm(i, 1);
        auto nn3 = get_norm(i, 2);

        x1[i] = nn1.x;
        y1[i] = nn1.y;
        z1[i] = nn1.z;

        x2[i] = nn2.x;
        y2[i] = nn2.y;
        z2[i] = nn2.z;

        x3[i] = nn3.x;
        y3[i] = nn3.y;
        z3[i] = nn3.z;
    }

    vector<3, T> n1(x1, y1, z1);
    vector<3, T> n2(x2, y2, z2);
    vector<3, T> n3(x3, y3, z3);

    return normalize( lerp(n1, n2, n3, hr.u, hr.v) );
}


//-------------------------------------------------------------------------------------------------
// Get normal from plane primitive
//

template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const& hr, basic_plane<3, T> const& plane)
{
    VSNRAY_UNUSED(hr);

    return plane.normal;
}


//-------------------------------------------------------------------------------------------------
// Get normal on sphere surface
//

template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const& hr, basic_sphere<T> const& sphere)
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
