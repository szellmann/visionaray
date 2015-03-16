// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_PATHTRACING_INL
#define VSNRAY_PATHTRACING_INL

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include <visionaray/generic_prim.h>
#include <visionaray/surface.h>

#include "traverse.h"

namespace visionaray
{
namespace pathtracing
{

template <typename Params>
struct kernel
{

    Params params;

    template <typename R, template <typename> class SMP>
    VSNRAY_FUNC vector<4, typename R::scalar_type> operator()(R ray, SMP<typename R::scalar_type>& s) const
    {

        using C = vector<4, typename R::scalar_type>;
        using S = typename R::scalar_type;
        using V = typename R::vec_type;

        /*static*/ const unsigned MaxDepth = 5;

        auto hit_rec        =  closest_hit(ray, params.prims.begin, params.prims.end);
        auto exited         = !hit_rec.hit;
        auto active_rays    =  hit_rec.hit;

        C result = select( exited, C(params.bg_color), C(1.0) );

        for (unsigned d = 0; d < MaxDepth; ++d)
        {
            if ( any(active_rays) )
            {
                V refl_dir;
                V view_dir = -ray.dir;

                auto surf = get_surface(hit_rec, params);
                auto below = active_rays & (dot(view_dir, surf.normal) < S(0.0));

                if (any(below))
                {
                    result = mul( result, C(0.0, 0.0, 0.0, 1.0), below, result );
                    active_rays = active_rays & !below;

                    if ( !any(active_rays) )
                    {
                        break;
                    }
                }


                S pdf(0.0);
                auto sr     = make_shade_record<Params, S>();
                sr.active   = hit_rec.hit;
                sr.view_dir = view_dir;
 
                auto color = surf.sample(sr, refl_dir, pdf, s);

                auto emissive = has_emissive_material(surf);
                color = mul( color, dot(surf.normal, refl_dir) / pdf, !emissive, color ); // TODO: maybe have emissive material return refl_dir so that dot(N,R) = 1?
                result = mul( result, C(color, S(1.0)), active_rays, result );
                active_rays &= !emissive;

                if (!any(active_rays))
                {
                    break;
                }

                auto isect_pos = ray.ori + ray.dir * hit_rec.t; // TODO: store in hit_rec?!?

                ray.ori = isect_pos + refl_dir * S(params.epsilon);
                ray.dir = refl_dir;

                hit_rec      = closest_hit(ray, params.prims.begin, params.prims.end);
                exited       = active_rays & !hit_rec.hit;
                active_rays &= hit_rec.hit;
            }

            result = mul( result, C(params.ambient_color), exited, result );
        }

        result = mul( result, C(0.0, 0.0, 0.0, 1.0), active_rays, result );

        return result;
    }
};

} // pathtracing
} // visionaray

#endif // VSNRAY_PATHTRACING_INL
