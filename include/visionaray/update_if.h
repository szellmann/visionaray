// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_UPDATE_IF_H
#define VSNRAY_UPDATE_IF_H 1

#include "detail/macros.h"
#include "math/aabb.h"
#include "math/intersect.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Utility functions that can be reimplemented for user-supplied hit records
// Used by the traversal functions (linear traversal, BVH traversal, etc.)
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// update_if()
//
// Update hit record <dst> with params from hit record <src> if <cond> is met
//

// default implementation for hit_record<ray, primitive<unsigned>>
template <typename HR, typename Cond>
VSNRAY_FUNC
void update_if(HR& dst, HR const& src, Cond const& cond)
{
    dst.hit        |= cond;
    dst.t           = select( cond, src.t, dst.t );
    dst.prim_id     = select( cond, src.prim_id, dst.prim_id );
    dst.geom_id     = select( cond, src.geom_id, dst.geom_id );
    dst.u           = select( cond, src.u, dst.u );
    dst.v           = select( cond, src.v, dst.v );
}


//-------------------------------------------------------------------------------------------------
// is_closer()
//
// Check if hit record <query> is valid and closer than hit record <reference> in ray's
// parameter space
//

// default implementation for hit_record<ray, primitive<unsigned>>
template <typename HR>
VSNRAY_FUNC
auto is_closer(HR const& query, HR const& reference)
    -> decltype(query.t < reference.t)
{
    return query.hit && query.t < reference.t;
}

// query is box, reference is general primitive

template <typename R, typename T, typename HR>
VSNRAY_FUNC
auto is_closer(hit_record<R, basic_aabb<T>> const& query, HR const& reference)
    -> decltype(query.tnear < reference.t)
{
    return query.hit && query.tnear < reference.t;
}


//-------------------------------------------------------------------------------------------------
// is_closer() overload with tmin and tmax
//

template <typename HR, typename T>
VSNRAY_FUNC
auto is_closer(HR const& query, HR const& reference, T const& tmin, T const& tmax)
    -> decltype(is_closer(query, reference))
{
    return is_closer(query, reference) && query.t >= tmin && query.t <= tmax;
}

// specialization for aabb
template <typename R, typename T, typename HR>
VSNRAY_FUNC
auto is_closer(
        hit_record<R, basic_aabb<T>> const& query,
        HR const& reference,
        typename R::scalar_type const& tmin,
        typename R::scalar_type const& tmax
        )
    -> decltype(is_closer(query, reference))
{
    return is_closer(query, reference) && query.tfar >= tmin && query.tnear <= tmax;
}


//-------------------------------------------------------------------------------------------------
// Condition types for passing update conditions around
//

struct is_closer_t
{
    template <typename HR1, typename HR2>
    VSNRAY_FUNC
    auto operator()(HR1 const& query, HR2 const& reference)
        -> decltype(is_closer(query, reference))
    {
        return is_closer(query, reference);
    }

    template <typename HR1, typename HR2, typename T>
    VSNRAY_FUNC
    auto operator()(HR1 const& query, HR2 const& reference, T const& tmin, T const& tmax)
        -> decltype(is_closer(query, reference, tmin, tmax))
    {
        return is_closer(query, reference, tmin, tmax);
    }
};

} // visionaray

#endif // VSNRAY_UPDATE_IF_H
