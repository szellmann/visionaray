// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

#include <visionaray/math/detail/math.h> // div_up
#include <visionaray/random_sampler.h>

#include "macros.h"
#include "sched_common.h"
#include "semaphore.h"

namespace visionaray
{

namespace detail
{

struct sync_params
{
    sync_params()
        : render_loop_exit(false)
    {
    }

    std::mutex mutex;
    std::condition_variable threads_start;
    visionaray::semaphore   threads_ready;

    std::atomic<long>       tile_idx_counter;
    std::atomic<long>       tile_fin_counter;
    std::atomic<long>       tile_num;

    std::atomic<bool>       render_loop_exit;
};


} // detail


//-------------------------------------------------------------------------------------------------
// Private implementation
//

template <typename R>
struct tiled_sched<R>::impl
{
    // TODO: any sampler
    typedef std::function<void(recti const&, random_sampler<typename R::scalar_type>&)> render_tile_func;

    void init_threads(unsigned num_threads);
    void destroy_threads();

    void render_loop();

    template <typename K, typename SP>
    void init_render_func(K kernel, SP sparams, unsigned frame_num);

    template <typename K, typename SP, typename Sampler, typename ...Args>
    void call_sample_pixel(
            std::false_type /* has intersector */,
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

    template <typename K, typename SP, typename Sampler, typename ...Args>
    void call_sample_pixel(
            std::true_type  /* has intersector */,
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

    std::vector<std::thread>    threads;
    detail::sync_params         sync_params;

    int                         width;
    int                         height;
    static const int            tile_width  = 16;
    static const int            tile_height = 16;
    recti                       scissor_box;

    render_tile_func            render_tile;
};

template <typename R>
void tiled_sched<R>::impl::init_threads(unsigned num_threads)
{
    for (unsigned i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([this](){ render_loop(); });
    }
}

template <typename R>
void tiled_sched<R>::impl::destroy_threads()
{
    if (threads.size() == 0)
    {
        return;
    }

    sync_params.render_loop_exit = true;
    sync_params.threads_start.notify_all();

    for (auto& t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    sync_params.render_loop_exit = false;
    threads.clear();
}

//-------------------------------------------------------------------------------------------------
// Main render loop
//

template <typename R>
void tiled_sched<R>::impl::render_loop()
{
    for (;;)
    {
        {
            std::unique_lock<std::mutex> l( sync_params.mutex );
            sync_params.threads_start.wait(l);
        }

    // case event.exit:
        if (sync_params.render_loop_exit)
        {
            break;
        }

    // case event.render:
        random_sampler<typename R::scalar_type> samp(detail::tic());
        for (;;)
        {
            auto tile_idx = sync_params.tile_idx_counter.fetch_add(1);

            if (tile_idx >= sync_params.tile_num)
            {
                break;
            }

            auto tilew = tile_width;
            auto tileh = tile_height;
            auto numtilesx = div_up( width, tilew );

            recti tile(
                    (tile_idx % numtilesx) * tilew,
                    (tile_idx / numtilesx) * tileh,
                    tilew,
                    tileh
                    );

            render_tile(tile, samp);

            auto num_tiles_fin = sync_params.tile_fin_counter.fetch_add(1);

            if (num_tiles_fin >= sync_params.tile_num - 1)
            {
                assert(num_tiles_fin == sync_params.tile_num - 1);
                sync_params.threads_ready.notify();
                break;
            }
        }
    }
}

template <typename R>
template <typename K, typename SP>
void tiled_sched<R>::impl::init_render_func(K kernel, SP sparams, unsigned frame_num)
{
    using T = typename R::scalar_type;

    width       = sparams.rt.width();
    height      = sparams.rt.height();
    scissor_box = sparams.scissor_box;

    recti clip_rect(scissor_box.x, scissor_box.y, scissor_box.w - 1, scissor_box.h - 1);

    render_tile = [=](recti const& tile, random_sampler<T>& samp)
    {
        unsigned numx = tile_width  / packet_size<T>::w;
        unsigned numy = tile_height / packet_size<T>::h;
        for (unsigned i = 0; i < numx * numy; ++i)
        {
            auto pos = vec2i(i % numx, i / numx);
            auto x = tile.x + pos.x * packet_size<T>::w;
            auto y = tile.y + pos.y * packet_size<T>::h;

            recti xpixel(x, y, packet_size<T>::w - 1, packet_size<T>::h - 1);
            if ( !overlapping(clip_rect, xpixel) )
            {
                continue;
            }

            call_sample_pixel(
                    typename detail::sched_params_has_intersector<SP>::type(),
                    kernel,
                    sparams,
                    samp,
                    frame_num,
                    x,
                    y,
                    sparams.rt.width(),
                    sparams.rt.height(),
                    sparams.cam
                    );
        }
    };
}



//-------------------------------------------------------------------------------------------------
// tiled_sched implementation
//

template <typename R>
tiled_sched<R>::tiled_sched(unsigned num_threads)
    : impl_(new impl())
{
    impl_->init_threads(num_threads);
}

template <typename R>
tiled_sched<R>::~tiled_sched()
{
    impl_->destroy_threads();
}

template <typename R>
template <typename K, typename SP>
void tiled_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.cam.begin_frame();

    sched_params.rt.begin_frame();

    impl_->init_render_func(kernel, sched_params, frame_num);

    auto numtilesx = div_up(impl_->width,  impl_->tile_width);
    auto numtilesy = div_up(impl_->height, impl_->tile_height);

    auto& sparams = impl_->sync_params;

    sparams.tile_idx_counter = 0;
    sparams.tile_fin_counter = 0;
    sparams.tile_num = numtilesx * numtilesy;

    // render frame
    sparams.threads_start.notify_all();

    sparams.threads_ready.wait();

    sched_params.rt.end_frame();

    sched_params.cam.end_frame();
}

template <typename R>
void tiled_sched<R>::reset(unsigned num_threads)
{
    if (static_cast<unsigned>(impl_->threads.size()) == num_threads)
    {
        return;
    }

    impl_->destroy_threads();
    impl_->init_threads(num_threads);
}

} // visionaray
