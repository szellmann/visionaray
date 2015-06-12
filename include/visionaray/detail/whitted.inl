// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_WHITTED_INL
#define VSNRAY_WHITTED_INL

#include <visionaray/result_record.h>
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

    template <typename Intersector, typename R>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(Intersector& isect, R ray) const
    {

        using S = typename R::scalar_type;
        using V = typename result_record<S>::vec_type;
        using C = spectrum<S>;

        result_record<S> result;

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end, isect);

        if (any(hit_rec.hit))
        {
            result.hit = hit_rec.hit;
            result.isect_pos = ray.ori + ray.dir * hit_rec.t;
        }
        else
        {
            result.hit = false;
            result.color = params.bg_color;
            return result;
        }

        C color(0.0);

        size_t depth = 0;
        auto no_hit_color = C(from_rgba(params.bg_color));
        auto mirror = S(1.0);
        while (any(hit_rec.hit) && any(mirror > S(params.epsilon)) && depth++ < params.num_bounces)
        {
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

            auto surf = get_surface(hit_rec, params);
            auto ambient = surf.material.ambient() * C(from_rgba(params.ambient_color));
            auto shaded_clr = select( hit_rec.hit, ambient, C(from_rgba(params.bg_color)) );
            auto view_dir = -ray.dir;

            auto n = surf.normal;

#if 1 // two-sided
            n = faceforward( n, view_dir, n );
#endif

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                auto light_dir = normalize( V(it->position()) - hit_rec.isect_pos );
                R shadow_ray
                (
                    hit_rec.isect_pos + light_dir * S(params.epsilon),
                    light_dir
                );

                // only cast a shadow if occluder between light source and hit pos
                auto shadow_rec  = any_hit(
                        shadow_ray,
                        params.prims.begin,
                        params.prims.end,
                        length(hit_rec.isect_pos - V(it->position())),
                        isect
                        );

                auto active_rays = hit_rec.hit & !shadow_rec.hit;

                auto sr         = make_shade_record<Params, S>();
                sr.active       = active_rays;
                sr.isect_pos    = hit_rec.isect_pos;
                sr.normal       = n;
                sr.view_dir     = -ray.dir;
                sr.light_dir    = light_dir;
                sr.light        = *it;
                auto clr        = surf.shade(sr);

                shaded_clr += select( active_rays, clr, C(0.0) );
            }

            color += select( hit_rec.hit, shaded_clr, no_hit_color ) * mirror;

            auto dir = reflect(view_dir, surf.normal);
            ray = R
            (
                hit_rec.isect_pos + dir * S(params.epsilon),
                dir
            );
            hit_rec = closest_hit(ray, params.prims.begin, params.prims.end, isect);
            mirror *= S(0.1);
            no_hit_color = C(0.0);
        }

        result.color = to_rgba(color);

        return result;

    }

    template <typename R>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(R ray) const
    {
        default_intersector ignore;
        return (*this)(ignore, ray);
    }
};

} // whitted
} // visionaray

#endif // VSNRAY_WHITTED_INL
