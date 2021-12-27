// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_PRIMITIVE_H
#define VSNRAY_GET_PRIMITIVE_H 1

#include <type_traits>

#include "detail/macros.h"
#include "bvh.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Get primitive from iterable list
//

template <
    typename Primitives,
    typename HR,
    typename P = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<!is_any_bvh<P>::value>::type
    >
VSNRAY_FUNC
inline P const& get_primitive(Primitives prims, HR const& hr)
{
    return prims[hr.prim_id];
}

// overload for BVHs
template <
    typename Primitives,
    typename HR,
    typename Base = typename HR::base_type,
    typename P = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<is_any_bvh<P>::value>::type,
    typename = typename std::enable_if<
                !is_any_bvh_inst<typename P::primitive_type>::value>::type // but not BVH of instances!
    >
VSNRAY_FUNC
inline typename P::primitive_type const& get_primitive(Primitives prims, HR const& hr)
{
    VSNRAY_UNUSED(hr);

    // TODO: iterate over list of BVHs and find the right one!
    return prims[0].primitive(hr.primitive_list_index);
}

// overload for BVHs of instances
template <
    typename Primitives,
    typename HR,
    typename Base = typename HR::base_type,
    typename P = typename std::iterator_traits<Primitives>::value_type,
    typename = typename std::enable_if<is_any_bvh<P>::value>::type, // is BVH ...
    typename = typename std::enable_if<
                is_any_bvh_inst<typename P::primitive_type>::value>::type, // ... of instances!
    typename X = void
    >
VSNRAY_FUNC
inline auto get_primitive(Primitives prims, HR const& hr)
    -> typename P::primitive_type::primitive_type const&
{
    // Assume we only have one top-level BVH (TODO?)
    return prims[0].primitive(hr.primitive_list_index).primitive(hr.primitive_list_index_inst);
}

} // visionaray

#endif // VSNRAY_GET_PRIMITIVE_H
