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

template <typename M, typename HR>
VSNRAY_FUNC
HR select(M const& m, HR const& hr1, HR const& hr2)
{
    HR result;
    result.hit     = select(m, hr1.hit, hr2.hit);
    result.t       = select(m, hr1.t, hr2.t);
    result.prim_id = select(m, hr1.prim_id, hr2.prim_id);
    result.geom_id = select(m, hr1.geom_id, hr2.geom_id);
    result.inst_id = select(m, hr1.inst_id, hr2.inst_id);
    result.u       = select(m, hr1.u, hr2.v);
    result.v       = select(m, hr1.u, hr2.v);
    return result;
}


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
    dst.hit     |= cond;
    dst.t        = select(cond, src.t, dst.t);
    dst.prim_id  = select(cond, src.prim_id, dst.prim_id);
    dst.geom_id  = select(cond, src.geom_id, dst.geom_id);
    dst.inst_id  = select(cond, src.inst_id, dst.inst_id);
    dst.u        = select(cond, src.u, dst.u);
    dst.v        = select(cond, src.v, dst.v);
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
{
    return query.hit && query.t < reference.t;
}

// query is box, reference is general primitive

template <typename R, typename T, typename HR>
VSNRAY_FUNC
auto is_closer(hit_record<R, basic_aabb<T>> const& query, HR const& reference)
{
    return query.hit && query.tnear < reference.t;
}


//-------------------------------------------------------------------------------------------------
// is_closer() overload with tmin and tmax
//

template <typename HR, typename T>
VSNRAY_FUNC
auto is_closer(HR const& query, HR const& reference, T const& tmin, T const& tmax)
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
    {
        return is_closer(query, reference);
    }

    template <typename HR1, typename HR2, typename T>
    VSNRAY_FUNC
    auto operator()(HR1 const& query, HR2 const& reference, T const& tmin, T const& tmax)
    {
        return is_closer(query, reference, tmin, tmax);
    }
};

} // visionaray

#endif // VSNRAY_UPDATE_IF_H
