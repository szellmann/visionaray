// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_WHITTED_INL
#define VSNRAY_WHITTED_INL

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

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);

        size_t depth = 0;
        auto color = C(0.0);
        auto no_hit_color = C(params.bg_color);
        auto mirror = S(1.0);
        while (any(hit_rec.hit) && any(mirror > S(params.epsilon)) && depth++ < 4/*1*/)
        {
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

            auto surf = get_surface(hit_rec, params);
            auto ambient = C(surf.material.ambient(), S(1.0)) * C(params.ambient_color);
            C shaded_clr = select( hit_rec.hit, ambient, C(params.bg_color) );

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                auto light_dir = normalize( V(it->position()) - hit_rec.isect_pos );
                R shadow_ray
                (
                    hit_rec.isect_pos + light_dir * S(params.epsilon),
                    light_dir
                );

                // only cast a shadow if occluder between light source and hit pos
                auto shadow_rec  = any_hit(shadow_ray, params.prims.begin, params.prims.end,
                    length(hit_rec.isect_pos - V(it->position())));
                auto active_rays = hit_rec.hit & !shadow_rec.hit;

                auto sr         = make_shade_record<Params, S>();
                sr.active       = active_rays;
                sr.isect_pos    = hit_rec.isect_pos;
                sr.view_dir     = -ray.dir;
                sr.light_dir    = light_dir;
                sr.light        = it;
                auto clr        = surf.shade(sr);

                shaded_clr += select( hit_rec.hit, C(clr, S(1.0)), C(0.0) );
            }

            color += select( hit_rec.hit, shaded_clr, no_hit_color ) * mirror;

            auto dir = reflect(ray.dir, surf.normal);
            ray = R
            (
                hit_rec.isect_pos + dir * S(params.epsilon),
                dir
            );
            hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);
            mirror *= S(0.1);
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
