// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CALL_KERNEL_H
#define VSNRAY_CALL_KERNEL_H

#include <utility>

#include <visionaray/kernels.h>

namespace visionaray
{

enum algorithm { Simple, Whitted, Pathtracing };

template <template <typename> class Kernel, typename Sched, typename SParams, typename KParams>
void call_kernel(Sched& sched, SParams const& sparams, KParams const& kparams, unsigned frame_num = 0)
{
    Kernel<KParams> kernel;
    kernel.params = kparams;
    sched.frame(kernel, sparams, frame_num);
}

template <typename Sched, typename KParams, typename ...Args>
void call_kernel(algorithm algo, Sched& sched, KParams const& kparams, unsigned& frame_num, Args&&... args)
{
    switch (algo)
    {

    case Simple:
        call_kernel<simple::kernel>
        (
            sched,
            make_sched_params<pixel_sampler::uniform_type>(std::forward<Args>(args)...),
            kparams
        );
        break;

    case Whitted:
        call_kernel<whitted::kernel>
        (
            sched,
            make_sched_params<pixel_sampler::uniform_type>(std::forward<Args>(args)...),
            kparams
        );
        break;

    case Pathtracing:
        call_kernel<pathtracing::kernel>
        (
            sched,
            make_sched_params<pixel_sampler::jittered_blend_type>(std::forward<Args>(args)...),
            kparams,
            ++frame_num
        );
        break;

    }
}

} // visionaray

#endif // VSNRAY_CALL_KERNEL_H
