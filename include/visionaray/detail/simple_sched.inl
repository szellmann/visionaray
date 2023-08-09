// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../make_generator.h"
#include "../make_random_seed.h"
#include "../packet_traits.h"

#include "sched_common.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Simple sched
//

template <typename R>
template <typename K, typename SP>
void simple_sched<R>::frame(K kernel, SP sched_params)
{
    using S = typename R::scalar_type;
    using I = typename simd::int_type<S>::type;

    sched_params.cam.begin_frame();

    sched_params.rt.begin_frame();


    for (int y = 0; y < sched_params.rt.height(); ++y)
    {
        for (int x = 0; x < sched_params.rt.width(); ++x)
        {
            expand_pixel<S> ep;
            auto seed = make_random_seed(
                convert_to_int(ep.y(y)) * sched_params.rt.width() + convert_to_int(ep.x(x)),
                I(frame_id_)
                );

            auto gen = make_generator(S{}, typename SP::pixel_sampler_type{}, seed);

            sample_pixel(
                    kernel,
                    typename SP::pixel_sampler_type{},
                    R{},
                    gen,
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

    ++frame_id_;
}

} // visionaray
