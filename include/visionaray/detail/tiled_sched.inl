// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

#include "../parallel_for.h"
#include "../random_sampler.h"
#include "sched_common.h"

namespace visionaray
{
namespace tiled_sched_impl
{

//-------------------------------------------------------------------------------------------------
// Generate primary ray and sample pixel
//

template <typename R, typename K, typename SP, typename Sampler, typename ...Args>
void call_sample_pixel(
        std::false_type /* has intersector */,
        R               /* */,
        K               kernel,
        SP              sparams,
        Sampler&        samp,
        unsigned        frame_num,
        Args&&...       args
        )
{
    auto r = detail::make_primary_rays(
            R{},
            typename SP::pixel_sampler_type{},
            samp,
            std::forward<Args>(args)...
            );

    sample_pixel(
            kernel,
            typename SP::pixel_sampler_type(),
            r,
            samp,
            frame_num,
            sparams.rt.ref(),
            std::forward<Args>(args)...
            );
}

template <typename R, typename K, typename SP, typename Sampler, typename ...Args>
void call_sample_pixel(
        std::true_type  /* has intersector */,
        R               /* */,
        K               kernel,
        SP              sparams,
        Sampler&        samp,
        unsigned        frame_num,
        Args&&...       args
        )
{
    auto r = detail::make_primary_rays(
            R{},
            typename SP::pixel_sampler_type{},
            samp,
            std::forward<Args>(args)...
            );

    sample_pixel(
            detail::have_intersector_tag(),
            sparams.intersector,
            kernel,
            typename SP::pixel_sampler_type(),
            r,
            samp,
            frame_num,
            sparams.rt.ref(),
            std::forward<Args>(args)...
            );
}

} // tiled_sched_impl


//-------------------------------------------------------------------------------------------------
// tiled_sched implementation
//

template <typename R>
tiled_sched<R>::tiled_sched(unsigned num_threads)
    : pool_(num_threads)
{
}

template <typename R>
template <typename K, typename SP>
void tiled_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.cam.begin_frame();

    sched_params.rt.begin_frame();

    random_sampler<typename R::scalar_type> samp(detail::tic(typename R::scalar_type{}));

    recti clip_rect(
            sched_params.scissor_box.x,
            sched_params.scissor_box.y,
            sched_params.scissor_box.w - 1,
            sched_params.scissor_box.h - 1
            );

    int w = sched_params.rt.width();
    int h = sched_params.rt.height();

    static const int dx = 16;
    static const int dy = 16;

    parallel_for(
        pool_,
        tiled_range2d<int>(0, w, dx, 0, h, dy),
        [&](range2d<int> const& r)
        {
            using T = typename R::scalar_type;

            int numx = dx / packet_size<T>::w;
            int numy = dy / packet_size<T>::h;
            for (int i = 0; i < numx * numy; ++i)
            {
                vec2i pos(i % numx, i / numx);
                int x = r.row_begin() + pos.x * packet_size<T>::w;
                int y = r.col_begin() + pos.y * packet_size<T>::h;

                recti xpixel(x, y, packet_size<T>::w - 1, packet_size<T>::h - 1);
                if ( !overlapping(clip_rect, xpixel) )
                {
                    continue;
                }

                tiled_sched_impl::call_sample_pixel(
                        typename detail::sched_params_has_intersector<SP>::type(),
                        R{},
                        kernel,
                        sched_params,
                        samp,
                        frame_num,
                        x,
                        y,
                        w,
                        h,
                        sched_params.cam
                        );
            }
        });

    sched_params.rt.end_frame();

    sched_params.cam.end_frame();
}

template <typename R>
void tiled_sched<R>::reset(unsigned num_threads)
{
    if (pool_.num_threads == num_threads)
    {
        return;
    }

    pool_.reset(num_threads);
}

} // visionaray
