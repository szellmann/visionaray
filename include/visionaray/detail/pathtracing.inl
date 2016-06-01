// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PATHTRACING_INL
#define VSNRAY_PATHTRACING_INL 1

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include <visionaray/get_surface.h>
#include <visionaray/result_record.h>
#include <visionaray/traverse.h>

namespace visionaray
{
namespace pathtracing
{

template <typename Params>
struct kernel
{

    Params params;

    template <typename Intersector, typename R, typename Sampler>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(
            Intersector& isect,
            R ray,
            Sampler& s
            ) const
    {

        using S = typename R::scalar_type;
        using V = typename result_record<S>::vec_type;
        using C = spectrum<S>;

        result_record<S> result;

        auto hit_rec        =  closest_hit(ray, params.prims.begin, params.prims.end, isect);
        auto exited         = !hit_rec.hit;
        auto active_rays    =  hit_rec.hit;
        result.color        =  params.bg_color;


        if (any(hit_rec.hit))
        {
            result.hit = hit_rec.hit;
            result.isect_pos = ray.ori + ray.dir * hit_rec.t;
        }
        else
        {
            result.hit = false;
            return result;
        }

        C dst(1.0);

        for (unsigned d = 0; d < params.num_bounces; ++d)
        {
            if ( any(active_rays) )
            {
                V refl_dir;
                V view_dir = -ray.dir;

                auto surf = get_surface(hit_rec, params);

                auto n = surf.shading_normal;

#if 1 // two-sided
                n = faceforward( n, view_dir, surf.normal );
#endif

                S pdf(0.0);
                auto sr     = make_shade_record<Params, S>();
                sr.active   = active_rays;
                sr.normal   = n;
                sr.view_dir = view_dir;
 
                auto src = surf.sample(sr, refl_dir, pdf, s);

                auto zero_pdf = pdf <= S(0.0);
                auto emissive = has_emissive_material(surf);

                src = mul( src, dot(n, refl_dir) / pdf, !emissive, src ); // TODO: maybe have emissive material return refl_dir so that dot(N,R) = 1?
                dst = mul( dst, src, active_rays && !zero_pdf, dst );
                dst = mul( dst, C(0.0), zero_pdf && active_rays, dst );

                active_rays &= !emissive;
                active_rays &= !zero_pdf;


                if (!any(active_rays))
                {
                    break;
                }

                auto isect_pos = ray.ori + ray.dir * hit_rec.t; // TODO: store in hit_rec?!?

                ray.ori = isect_pos + refl_dir * S(params.epsilon);
                ray.dir = refl_dir;

                hit_rec      = closest_hit(ray, params.prims.begin, params.prims.end, isect);
                exited       = active_rays & !hit_rec.hit;
                active_rays &= hit_rec.hit;
            }

            dst = mul( dst, C(from_rgba(params.ambient_color)), exited, dst );
        }

        dst = mul( dst, C(0.0), active_rays, dst );

        result.color = select( result.hit, to_rgba(dst), result.color );

        return result;
    }

    template <typename R, typename Sampler>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(
            R ray,
            Sampler& s
            ) const 
    {
        default_intersector ignore;
        return (*this)(ignore, ray, s);
    }
};

} // pathtracing
} // visionaray

#endif // VSNRAY_PATHTRACING_INL
