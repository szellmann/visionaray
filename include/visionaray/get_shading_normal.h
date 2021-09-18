// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_SHADING_NORMAL_H
#define VSNRAY_GET_SHADING_NORMAL_H 1

#include <iterator>
#include <type_traits>

#include "detail/macros.h"
#include "math/detail/math.h"
#include "math/simd/type_traits.h"
#include "math/triangle.h"
#include "get_normal.h"
#include "prim_traits.h"
#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Default get_shading_normal w/o normal list, dispatches to get_normal (geometric!)
//

template <
    typename HR,
    typename Primitive,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_shading_normal(HR const& hr, Primitive prim)
    -> decltype(get_normal(hr, prim))
{
    return get_normal(hr, prim);
}


//-------------------------------------------------------------------------------------------------
// Default get_shading_normal implementation, assumes that normal list is unused!
//

template <
    typename Normals,
    typename HR,
    typename Primitive,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_shading_normal(
        Normals       normals,
        HR const&     hr,
        Primitive     prim,
        NormalBinding binding
        )
{
    VSNRAY_UNUSED(normals);
    VSNRAY_UNUSED(binding);

    return get_shading_normal(hr, prim);
}

template <
    typename HR,
    typename Primitive
    >
VSNRAY_FUNC
inline auto get_shading_normal(HR const& hr, Primitive prim)
    -> decltype( get_normal(
            hr,
            prim
            ) )
{
    return get_normal(hr, prim);
}


//-------------------------------------------------------------------------------------------------
// get_shading_normal for triangles with normals_per_vertex_binding
//

template <
    typename Normals,
    typename HR,
    typename T,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_shading_normal(
        Normals                     normals,
        HR const&                   hr,
        basic_triangle<3, T>        /* */,
        normals_per_vertex_binding  /* */
        )
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
// get_shading_normal for triangles with normals_per_vertex_binding for SIMD ray
//

template <
    typename Normals,
    typename HR,
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_shading_normal(
        Normals                    normals,
        HR const&                  hr,
        basic_triangle<3, T>       /* */,
        normals_per_vertex_binding /* */
        )
    -> vector<3, typename HR::scalar_type>
{
    using U = typename HR::scalar_type;
    using N = typename std::iterator_traits<Normals>::value_type;
    using float_array = simd::aligned_array_t<U>;

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

    for (unsigned i = 0; i < simd::num_elements<U>::value; ++i)
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

    vector<3, U> n1(x1, y1, z1);
    vector<3, U> n2(x2, y2, z2);
    vector<3, U> n3(x3, y3, z3);

    return normalize( lerp(n1, n2, n3, hr.u, hr.v) );
}

} // visionaray

#endif // VSNRAY_GET_SHADING_NORMAL_H
