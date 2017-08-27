// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PATHTRACING_INL
#define VSNRAY_DETAIL_PATHTRACING_INL 1

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

        simd::mask_type_t<S> active_rays = true;

        C dst(1.0);

        result_record<S> result;
        result.color = params.bg_color;

        for (unsigned bounce = 0; bounce < params.num_bounces; ++bounce)
        {
            auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end, isect);

            // Handle rays that just exited
            auto exited = active_rays & !hit_rec.hit;
            dst = mul( dst, C(from_rgba(params.ambient_color)), exited, dst );


            // Exit if no ray is active anymore
            active_rays &= hit_rec.hit;

            if (!any(active_rays))
            {
                break;
            }

            // Special handling for first bounce
            if (bounce == 0)
            {
                result.hit = hit_rec.hit;
                result.isect_pos = ray.ori + ray.dir * hit_rec.t;
            }


            // Process the current bounce

            V refl_dir;
            V view_dir = -ray.dir;

            auto surf = get_surface(hit_rec, params);

            auto n = surf.shading_normal;

#if 1 // two-sided
            n = faceforward( n, view_dir, surf.geometric_normal );
#endif

            S pdf(0.0);
            auto sr     = make_shade_record<Params, S>();
            sr.normal   = n;
            sr.view_dir = view_dir;

            auto src = surf.sample(sr, refl_dir, pdf, s);

            auto zero_pdf = pdf <= S(0.0);
            auto emissive = has_emissive_material(surf);

            src = mul( src, dot(n, refl_dir) / pdf, !emissive, src ); // TODO: maybe have emissive material return refl_dir so that dot(N,R) = 1?
            dst = mul( dst, src, active_rays && !zero_pdf, dst );
            dst = select( zero_pdf && active_rays, C(0.0), dst );

            active_rays &= !emissive;
            active_rays &= !zero_pdf;


            if (!any(active_rays))
            {
                break;
            }

            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

            ray.ori = hit_rec.isect_pos + refl_dir * S(params.epsilon);
            ray.dir = refl_dir;
        }

        // Terminate paths that are still active
        dst = select(active_rays, C(0.0), dst);

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

#endif // VSNRAY_DETAIL_PATHTRACING_INL
