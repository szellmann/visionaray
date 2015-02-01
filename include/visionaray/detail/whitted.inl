// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_WHITTED_INL
#define VSNRAY_WHITTED_INL

#include <limits>

#include <visionaray/surface.h>

#include "traverse.h"

namespace visionaray
{
namespace whitted
{

template <typename C, typename Params>
struct kernel
{

    Params params;

    template <typename R>
    VSNRAY_FUNC C operator()(R ray) const
    {

        typedef R ray_type;
        typedef typename R::scalar_type scalar_type;
        typedef typename R::vec_type vec_type;

        /*static*/ const scalar_type scene_epsilon(0.001);

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);

        C bg_color
        (
            scalar_type(0.1),
            scalar_type(0.4),
            scalar_type(1.0),
            scalar_type(1.0)
        );

        size_t depth = 0;
        auto color = C(0.0);
        auto no_hit_color = bg_color;
        auto mirror = scalar_type(1.0);
        while (any(hit_rec.hit) && any(mirror > scene_epsilon) && depth++ < 4/*1*/)
        {
            C shaded_clr(scalar_type(0.0));
            auto surf = get_surface(params, hit_rec);

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

                auto light_pos = normalize( vec_type(it->position_) );
                ray_type shadow_ray
                (
                    hit_rec.isect_pos + light_pos * scene_epsilon,
                    light_pos
                );

                auto shadow_rec  = any_hit(shadow_ray, params.prims.begin, params.prims.end);
                auto active_rays = hit_rec.hit & !shadow_rec.hit;

                auto sr     = make_shade_record<Params, scalar_type>();
                sr.active   = active_rays;
                sr.view_dir = -ray.dir;
                sr.light    = it;
                auto clr    = surf.shade(sr);

                shaded_clr += select( hit_rec.hit, C(clr, scalar_type(1.0)), C(scalar_type(0.0)) );
            }

            color += select( hit_rec.hit, shaded_clr, no_hit_color ) * mirror;

            auto negn = dot(ray.dir, surf.normal) > scalar_type(0.0);
            vec_type n = select(negn, -surf.normal, surf.normal);

            auto dir = reflect(ray.dir, n);
            ray = ray_type
            (
                hit_rec.isect_pos + dir * scene_epsilon,
                dir
            );
            hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);
            mirror *= scalar_type(0.8);
            no_hit_color = C(0.0);
        }

        if (depth == 0)
        {
            return bg_color;
        }
        else
        {
            return color;
        }

    }
};

} // whitted
} // visionaray

#endif // VSNRAY_WHITTED_INL


