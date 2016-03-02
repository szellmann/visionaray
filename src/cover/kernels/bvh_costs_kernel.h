// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COVER_BVH_COSTS_KERNEL_H
#define VSNRAY_COVER_BVH_COSTS_KERNEL_H 1

#include <visionaray/intersector.h>
#include <visionaray/result_record.h>
#include <visionaray/traverse.h>

namespace visionaray { namespace cover {

//-------------------------------------------------------------------------------------------------
// Intersector to gather bvh costs
//

struct bvh_cost_intersector : basic_intersector<bvh_cost_intersector>
{
    using basic_intersector<bvh_cost_intersector>::operator();

    template <typename R, typename S, typename ...Args>
    VSNRAY_FUNC
    auto operator()(R const& ray, basic_aabb<S> const& box, Args&&... args)
        -> decltype( intersect(ray, box, std::forward<Args>(args)...) )
    {
        ++num_boxes;
        return intersect(ray, box, std::forward<Args>(args)...);
    }

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(R const& ray, basic_triangle<3, S> const& tri)
        -> decltype( intersect(ray, tri) )
    {
        ++num_tris;
        return intersect(ray, tri);
    }

    unsigned num_boxes = 0;
    unsigned num_tris  = 0;
};


//-------------------------------------------------------------------------------------------------
// BVH costs debug kernel
//

template <typename Params>
struct bvh_costs_kernel
{
    VSNRAY_FUNC explicit bvh_costs_kernel(Params const& p)
        : params(p)
    {
    }

    template <typename R>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;


        // weights for box costs and primitive costs, in [0..1]
        S wb = 1.0f;
        S wp = 1.0f;

        result_record<S> result;

        bvh_cost_intersector i;

        auto hit_rec        = closest_hit(ray, params.prims.begin, params.prims.end, i);

        S t                 = i.num_boxes * wb + i.num_tris * wp;
        auto rgb            = temperature_to_rgb(t / S(120.0)); // plot max. 120 ray interactions..

        result.hit          = hit_rec.hit;
        result.color        = select( hit_rec.hit, C(rgb, S(1.0)), C(0.0) );
        result.isect_pos    = ray.ori + ray.dir * hit_rec.t;
        return result;
    }

    Params params;
};

}} // namespace visionaray::cover

#endif // VSNRAY_COVER_BVH_COSTS_KERNEL_H
