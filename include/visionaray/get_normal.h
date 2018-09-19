// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_NORMAL_H
#define VSNRAY_GET_NORMAL_H 1

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "math/simd/type_traits.h"
#include "math/plane.h"
#include "math/ray.h"
#include "math/sphere.h"
#include "math/triangle.h"
#include "math/vector.h"
#include "prim_traits.h"
#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Get face normal from array
//

// TODO: generalize this to primitives with num_normals<Primitive, NormalBinding>::value == 1 ?

template <
    typename Normals,
    typename HR,
    typename T,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        basic_triangle<3, T>        /* */,
        normals_per_face_binding    /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    return normals[hr.prim_id];
}

//-------------------------------------------------------------------------------------------------
// Gather N face normals for SIMD ray
//

template <
    typename Normals,
    typename HR,
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type,
    typename = void
    >
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        basic_triangle<3, T>        /* */,
        normals_per_face_binding    /* */
        )
    -> vector<3, typename HR::scalar_type>
{
    using U = typename HR::scalar_type;
    using N = typename std::iterator_traits<Normals>::value_type;
    using float_array = simd::aligned_array_t<U>;

    auto hrs = unpack(hr);

    float_array x;
    float_array y;
    float_array z;

    for (size_t i = 0; i < simd::num_elements<U>::value; ++i)
    {
        auto n = hrs[i].hit ? normals[hrs[i].prim_id] : N();
        x[i] = n.x;
        y[i] = n.y;
        z[i] = n.z;
    }

    return vector<3, U>(
            U(x),
            U(y),
            U(z)
            );
}


//-------------------------------------------------------------------------------------------------
// Dispatch to calculating function - if normals are not bound per face
//

template <typename Normals, typename HR, typename T>
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        basic_triangle<3, T>        prim,
        normals_per_vertex_binding  /* */
        )
    -> decltype( get_normal(hr, prim) )
{
    VSNRAY_UNUSED(normals);

    return get_normal(hr, prim);
}


//-------------------------------------------------------------------------------------------------
// Get normal from triangle primitive
//

template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const& hr, basic_triangle<3, T> const& triangle)
{
    VSNRAY_UNUSED(hr);

    return normalize(cross(triangle.e1, triangle.e2));
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
// get_normal as functor for template arguments
//

namespace detail
{

struct get_normal_t
{
    template <typename Normals, typename HR, typename Primitive, typename NormalBinding>
    VSNRAY_FUNC
    inline auto operator()(
            Normals                     normals,
            HR const&                   hr,
            Primitive                   prim,
            NormalBinding               /* */
            ) const
        -> decltype( get_normal(normals, hr, prim, NormalBinding{}) )
    {
        return get_normal(normals, hr, prim, NormalBinding{});
    }

    template <typename HR, typename Primitive>
    VSNRAY_FUNC
    inline auto operator()(
            HR const&                   hr,
            Primitive                   prim
            ) const
        -> decltype( get_normal(hr, prim) )
    {
        return get_normal(hr, prim);
    }
};

} // detail
} // visionaray

#endif // VSNRAY_GET_NORMAL_H
