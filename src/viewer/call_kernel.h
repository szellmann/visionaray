// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_CALL_KERNEL_H
#define VSNRAY_VIEWER_CALL_KERNEL_H 1

#include <utility>

#include <visionaray/kernels.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>
#include <visionaray/tags.h>
#include <visionaray/thin_lens_camera.h>
#include <visionaray/variant.h>

#include "bvh_costs.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Convenience functions to call built-in or custom kernels with a scheduler
//
// This basically wraps sched.frame()
// Determines pixel-sampler types based on the algorithm chosen
// Handles SSAA
//
//-------------------------------------------------------------------------------------------------



enum algorithm { Simple, Whitted, Pathtracing, Costs };


//-------------------------------------------------------------------------------------------------
// Pinhole camera vs. thin lens camera
//

template <typename Sched, typename KParams, typename RT>
inline void call_kernel(
        algorithm                                        algo,
        Sched&                                           sched,
        KParams const&                                   kparams,
        unsigned&                                        frame_num,
        unsigned                                         spp,
        variant<pinhole_camera, thin_lens_camera> const& cam,
        RT&                                              rt
        )
{
    if (cam.as<thin_lens_camera>())
    {
        call_kernel(
                algo,
                sched,
                kparams,
                frame_num,
                spp,
                *cam.as<thin_lens_camera>(),
                rt
                );
    }
    else
    {
        call_kernel(
                algo,
                sched,
                kparams,
                frame_num,
                spp,
                *cam.as<pinhole_camera>(),
                rt
                );
    }
}


//-------------------------------------------------------------------------------------------------
// Call one of the built-in kernels
//

template <typename Sched, typename KParams, typename ...Args>
void call_kernel(
        algorithm       algo,
        Sched&          sched,
        KParams const&  kparams,
        unsigned&       frame_num,
        unsigned        spp,
        Args&&...       args
        )
{
    switch (algo)
    {

    case Simple:
    {
        pixel_sampler::uniform_type ups;
        ups.ssaa_factor = spp;
        sched.frame(
            simple::kernel<KParams>({kparams}),
            make_sched_params(ups, std::forward<Args>(args)...)
            );
        break;
    }
    case Whitted:
    {
        pixel_sampler::uniform_type ups;
        ups.ssaa_factor = spp;
        sched.frame(
            whitted::kernel<KParams>({kparams}),
            make_sched_params(ups, std::forward<Args>(args)...)
            );
        break;
    }
    case Pathtracing:
    {
        float alpha = 1.0f / ++frame_num;
        pixel_sampler::jittered_blend_type blend_params;
        blend_params.sfactor = alpha;
        blend_params.dfactor = 1.0f - alpha;
        sched.frame(
            pathtracing::kernel<KParams>({kparams}),
            make_sched_params(blend_params, std::forward<Args>(args)...)
            );
        break;
    }
    case Costs:
    {
        sched.frame(
            bvh_costs_kernel<KParams>({kparams}),
            make_sched_params(pixel_sampler::uniform_type{}, std::forward<Args>(args)...)
            );
        break;
    }
    }
}

} // visionaray

#endif // VSNRAY_VIEWER_CALL_KERNEL_H
