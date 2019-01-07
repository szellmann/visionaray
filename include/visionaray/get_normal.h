// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_NORMAL_H
#define VSNRAY_GET_NORMAL_H 1

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "detail/macros.h"
#include "math/simd/type_traits.h"
#include "math/plane.h"
#include "math/sphere.h"
#include "math/triangle.h"
#include "math/vector.h"
#include "prim_traits.h"
#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Default get_normal implementation, assumes that normal list is unused!
//

template <typename Normals, typename HR, typename Primitive>
VSNRAY_FUNC
inline auto get_normal(Normals normals, HR const& hr, Primitive const& prim)
    -> typename std::iterator_traits<Normals>::value_type
{
    VSNRAY_UNUSED(normals);

    return get_normal(hr, prim);
}


//-------------------------------------------------------------------------------------------------
// Get face normal from array
//

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
        basic_triangle<3, T>        /* */
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
        basic_triangle<3, T>        /* */
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

} // visionaray

#endif // VSNRAY_GET_NORMAL_H
