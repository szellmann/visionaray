// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CALL_KERNEL_H
#define VSNRAY_CALL_KERNEL_H 1

#include <utility>

#include <visionaray/kernels.h>

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
// Call custom kernels
//

template <template <typename> class Kernel, typename Sched, typename SParams, typename KParams>
void call_kernel(Sched& sched, SParams const& sparams, KParams const& kparams, unsigned frame_num = 0)
{
    Kernel<KParams> kernel;
    kernel.params = kparams;
    sched.frame(kernel, sparams, frame_num);
}


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
            call_kernel<simple::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<1>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        else if (ssaa_samples == 2)
        {
            call_kernel<simple::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<2>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        else if (ssaa_samples == 4)
        {
            call_kernel<simple::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<4>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        else if (ssaa_samples == 8)
        {
            call_kernel<simple::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<8>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        break;

    case Whitted:
        if (ssaa_samples == 1)
        {
            call_kernel<whitted::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<1>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        else if (ssaa_samples == 2)
        {
            call_kernel<whitted::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<2>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        else if (ssaa_samples == 4)
        {
            call_kernel<whitted::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<4>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        else if (ssaa_samples == 8)
        {
            call_kernel<whitted::kernel>(
                sched,
                make_sched_params(pixel_sampler::ssaa_type<8>{}, std::forward<Args>(args)...),
                kparams
                );
        }
        break;

    case Pathtracing:
        call_kernel<pathtracing::kernel>(
            sched,
            make_sched_params(pixel_sampler::jittered_blend_type{}, std::forward<Args>(args)...),
            kparams,
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
        call_kernel<simple::kernel>(
            sched,
            make_sched_params(pixel_sampler::uniform_type{}, std::forward<Args>(args)...),
            kparams
            );
        break;

    case Whitted:
        call_kernel<whitted::kernel>(
            sched,
            make_sched_params(pixel_sampler::uniform_type{}, std::forward<Args>(args)...),
            kparams
            );
        break;

    case Pathtracing:
        call_kernel<pathtracing::kernel>(
            sched,
            make_sched_params(pixel_sampler::jittered_blend_type{}, std::forward<Args>(args)...),
            kparams,
            ++frame_num
            );
        break;

    }
}

} // visionaray

#endif // VSNRAY_CALL_KERNEL_H
