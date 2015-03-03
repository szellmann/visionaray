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

template <typename Params>
struct kernel
{

    Params params;

    template <typename R>
    VSNRAY_FUNC vector<4, typename R::scalar_type> operator()(R ray) const
    {

        using C = vector<4, typename R::scalar_type>;
        using S = typename R::scalar_type;
        using V = typename R::vec_type;

        /*static*/ const S scene_epsilon(0.001);

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);

        size_t depth = 0;
        auto color = C(0.0);
        auto no_hit_color = C(params.bg_color);
        auto mirror = S(1.0);
        while (any(hit_rec.hit) && any(mirror > scene_epsilon) && depth++ < 4/*1*/)
        {
            auto surf = get_surface(hit_rec, params);
            C shaded_clr = select( hit_rec.hit, C(surf.material.ambient(), S(1.0)), C(params.bg_color) );

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

                auto light_dir = normalize( V(it->position()) );
                R shadow_ray
                (
                    hit_rec.isect_pos + light_dir * scene_epsilon,
                    light_dir
                );

                auto shadow_rec  = any_hit(shadow_ray, params.prims.begin, params.prims.end);
                auto active_rays = hit_rec.hit & !shadow_rec.hit;

                auto sr     = make_shade_record<Params, S>();
                sr.active   = active_rays;
                sr.view_dir = -ray.dir;
                sr.light    = it;
                auto clr    = surf.shade(sr);

                shaded_clr += select( hit_rec.hit, C(clr, S(1.0)), C(0.0) );
            }

            color += select( hit_rec.hit, shaded_clr, no_hit_color ) * mirror;

            auto negn = dot(ray.dir, surf.normal) > S(0.0);
            V n = select(negn, -surf.normal, surf.normal);

            auto dir = reflect(ray.dir, n);
            ray = R
            (
                hit_rec.isect_pos + dir * scene_epsilon,
                dir
            );
            hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);
            mirror *= S(0.5);
            no_hit_color = C(0.0);
        }

        if (depth == 0)
        {
            return C(params.bg_color);
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


