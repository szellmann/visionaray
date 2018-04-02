// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/random_generator.h>

#include "sched_common.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Simple sched
//

template <typename R>
template <typename K, typename SP>
void simple_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.cam.begin_frame();

    sched_params.rt.begin_frame();


    auto scissor_box = sched_params.scissor_box;

    for (int y = 0; y < sched_params.rt.height(); ++y)
    {
        for (int x = 0; x < sched_params.rt.width(); ++x)
        {
            if (x < scissor_box.x || y < scissor_box.y || x >= scissor_box.w || y >= scissor_box.h)
            {
                continue;
            }

            // TODO: support any generator
            random_generator<typename R::scalar_type> samp(detail::tic(typename R::scalar_type{}));

            auto r = detail::make_primary_rays(
                    R{},
                    typename SP::pixel_sampler_type{},
                    samp,
                    x,
                    y,
                    sched_params.rt.width(),
                    sched_params.rt.height(),
                    sched_params.cam
                    );

            sample_pixel(
                    kernel,
                    typename SP::pixel_sampler_type{},
                    r,
                    samp,
                    frame_num,
                    sched_params.rt.ref(),
                    x,
                    y,
                    sched_params.rt.width(),
                    sched_params.rt.height(),
                    sched_params.cam
                    );
        }
    }


    sched_params.rt.end_frame();

    sched_params.cam.end_frame();
}

} // visionaray
