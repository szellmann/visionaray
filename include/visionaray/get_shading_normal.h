// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_SHADING_NORMAL
#define VSNRAY_GET_SHADING_NORMAL 1

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "detail/macros.h"
#include "math/detail/math.h"
#include "math/simd/type_traits.h"
#include "math/intersect.h"
#include "math/primitive.h"
#include "math/ray.h"
#include "math/triangle.h"
#include "get_normal.h"
#include "prim_traits.h"
#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Default get_shading_normal implementation - simply use the geometric normal for shading
//

template <
    typename Normals,
    typename HR,
    typename Primitive,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_shading_normal(
        Normals                     normals,
        HR const&                   hr,
        Primitive                   prim,
        NormalBinding               binding
        )
    -> decltype( get_normal(
            normals,
            hr,
            prim,
            binding
            ) )
{
    return get_normal(normals, hr, prim, binding);
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

template <typename Normals, typename HR, typename T>
VSNRAY_FUNC
inline auto get_shading_normal(
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
// get_shading_normal for triangles with normals_per_vertex_binding for SIMD ray
//

template <
    typename Normals,
    typename T,
    typename U,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_FUNC
inline vector<3, T> get_shading_normal(
        Normals                                                 normals,
        hit_record<basic_ray<T>, primitive<unsigned>> const&    hr,
        basic_triangle<3, U>                                    /* */,
        normals_per_vertex_binding                              /* */
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using float_array = simd::aligned_array_t<T>;

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
// get_shading_normal as functor for template arguments
//

namespace detail
{

struct get_shading_normal_t
{
    template <typename Normals, typename HR, typename Primitive, typename NormalBinding>
    VSNRAY_FUNC
    inline auto operator()(
            Normals                     normals,
            HR const&                   hr,
            Primitive                   prim,
            NormalBinding               /* */
            ) const
        -> decltype( get_shading_normal(normals, hr, prim, NormalBinding{}) )
    {
        return get_shading_normal(normals, hr, prim, NormalBinding{});
    }

    template <typename HR, typename Primitive>
    VSNRAY_FUNC
    inline auto operator()(
            HR const&                   hr,
            Primitive                   prim
            ) const
        -> decltype( get_shading_normal(hr, prim) )
    {
        return get_shading_normal(hr, prim);
    }
};

} // detail
} // visionaray

#endif // VSNRAY_GET_SHADING_NORMAL
