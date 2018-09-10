// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_AREA_H
#define VSNRAY_GET_AREA_H 1

#include <iterator>
#include <type_traits>
#include <utility>

#include "detail/macros.h"
#include "math/simd/type_traits.h"
#include "bvh.h"

namespace visionaray
{

#if 1
// TODO!!
template <
    typename P,
    typename = typename std::enable_if<is_any_bvh<P>::value>::type,
    typename Generator,
    typename U = typename Generator::value_type
    >
VSNRAY_FUNC
inline vector<3, U> sample_surface(P, Generator&)
{
    return {};
}
#endif

// No BVH, no SIMD
template <
    typename Primitives,
    typename HR,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<!is_any_bvh<Primitive>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_area(Primitives const& prims, HR const& hr)
    -> decltype(area(std::declval<Primitive>()))
{
    return area(prims[hr.prim_id]);
}

// No BVH, SIMD
template <
    typename Primitives,
    typename HR,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<!is_any_bvh<Primitive>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type,
    typename = void
    >
VSNRAY_FUNC
inline auto get_area(Primitives const& prims, HR const& hr)
    -> typename HR::scalar_type
{
    using T = typename HR::scalar_type;
    using float_array = simd::aligned_array_t<T>;
    using int_array = simd::aligned_array_t<simd::int_type_t<T>>;

    int_array prim_id;
    store(prim_id, hr.prim_id);

    float_array result = {};

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        result[i] = area(prims[prim_id[i]]);
    }

    return T(result);
}

// BVH, no SIMD
template <
    typename Primitives,
    typename HR,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type,
    typename = void,
    typename = void
    >
VSNRAY_FUNC
inline auto get_area(Primitives const& prims, HR const& hr)
    -> decltype(area(std::declval<typename Primitive::primitive_type>()))
{
    // Find the BVH that contains prim_id
    size_t num_primitives_total = 0;

    size_t i = 0;
    while (static_cast<size_t>(hr.prim_id) >= num_primitives_total + prims[i].num_primitives())
    {
        num_primitives_total += prims[i++].num_primitives();
    }

    return area(prims[i].primitive(hr.primitive_list_index));
}

// BVH, SIMD
template <
    typename Primitives,
    typename HR,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value && !is_any_bvh_inst<typename Primitive::primitive_type>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_area(Primitives const& prims, HR const& hr)
    -> typename HR::scalar_type
{
    using T = typename HR::scalar_type;
    using float_array = simd::aligned_array_t<T>;
    using int_array = simd::aligned_array_t<simd::int_type_t<T>>;

    int_array prim_id;
    store(prim_id, hr.prim_id);

    int_array primitive_list_index;
    store(primitive_list_index, hr.primitive_list_index);

    float_array result = {};

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        // Find the BVH that contains prim_id[i]
        size_t num_primitives_total = 0;

        size_t j = 0;
        while (static_cast<size_t>(prim_id[i]) >= num_primitives_total + prims[j].num_primitives())
        {
            num_primitives_total += prims[j++].num_primitives();
        }
        result[i] = area(prims[j].primitive(primitive_list_index[i]));
    }

    return T(result);
}

// BVH instance, SIMD
template <
    typename Primitives,
    typename HR,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value && is_any_bvh_inst<typename Primitive::primitive_type>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_area(Primitives const& prims, HR const& hr, void* = nullptr)
    -> typename HR::scalar_type
{
    using T = typename HR::scalar_type;
    using float_array = simd::aligned_array_t<T>;
    using int_array = simd::aligned_array_t<simd::int_type_t<T>>;

    int_array prim_id;
    store(prim_id, hr.prim_id);

//  int_array inst_id;
//  store(inst_id, hr.inst_id);
    int_array primitive_list_index;
    store(primitive_list_index, hr.primitive_list_index);

    float_array result = {};

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto& b = prims[0]; // TODO: currently only two levels supported (i.e. one top-level BVH)

        auto& inst = b.primitive(primitive_list_index[i]);

        result[i] = area(inst.primitive(prim_id[i]));
    }

    return T(result);
}

} // visionaray

#endif // VSNRAY_GET_AREA_H
