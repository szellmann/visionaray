// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_CALL_KERNEL_H
#define VSNRAY_VIEWER_CALL_KERNEL_H 1

#include <utility>

#include <visionaray/kernels.h>
#include <visionaray/scheduler.h>
#include <visionaray/tags.h>

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



enum algorithm { Simple, Whitted, Pathtracing };


//-------------------------------------------------------------------------------------------------
// Call one of the built-in kernels
//
// Simple, Whitted: mind ssaa_samples
// Pathtracing:     jittered-blend sampling
//

template <typename Sched, typename KParams, typename ...Args>
void call_kernel(
        algorithm       algo,
        Sched&          sched,
        KParams const&  kparams,
        unsigned&       frame_num,
        unsigned        ssaa_samples,
        Args&&...       args
        )
{
    switch (algo)
    {

    case Simple:
        if (ssaa_samples == 1)
        {
            sched.frame(
                simple::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<1>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        else if (ssaa_samples == 2)
        {
            sched.frame(
                simple::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<2>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        else if (ssaa_samples == 4)
        {
            sched.frame(
                simple::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<4>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        else if (ssaa_samples == 8)
        {
            sched.frame(
                simple::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<8>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        break;

    case Whitted:
        if (ssaa_samples == 1)
        {
            sched.frame(
                whitted::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<1>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        else if (ssaa_samples == 2)
        {
            sched.frame(
                whitted::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<2>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        else if (ssaa_samples == 4)
        {
            sched.frame(
                whitted::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<4>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        else if (ssaa_samples == 8)
        {
            sched.frame(
                whitted::kernel<KParams>({kparams}),
                make_sched_params(pixel_sampler::ssaa_type<8>{}, std::forward<Args>(args)...),
                frame_num
                );
        }
        break;

    case Pathtracing:
        sched.frame(
            pathtracing::kernel<KParams>({kparams}),
            make_sched_params(pixel_sampler::jittered_blend_type{}, std::forward<Args>(args)...),
            ++frame_num
            );
        break;

    }
}


//-------------------------------------------------------------------------------------------------
// Call one of the built-in kernels
//
// Simple, Whitted: uniform sampling (1x SSAA)
// Pathtracing:     jittered-blend sampling
//

template <typename Sched, typename KParams, typename ...Args>
void call_kernel(
        algorithm       algo,
        Sched&          sched,
        KParams const&  kparams,
        unsigned&       frame_num,
        Args&&...       args
        )
{
    switch (algo)
    {

    case Simple:
        sched.frame(
            simple::kernel<KParams>({kparams}),
            make_sched_params(pixel_sampler::uniform_type{}, std::forward<Args>(args)...),
            frame_num
            );
        break;

    case Whitted:
        sched.frame(
            whitted::kernel<KParams>({kparams}),
            make_sched_params(pixel_sampler::uniform_type{}, std::forward<Args>(args)...),
            frame_num
            );
        break;

    case Pathtracing:
        sched.frame(
            pathtracing::kernel<KParams>({kparams}),
            make_sched_params(pixel_sampler::jittered_blend_type{}, std::forward<Args>(args)...),
            ++frame_num
            );
        break;

    }
}

} // visionaray

#endif // VSNRAY_VIEWER_CALL_KERNEL_H
