// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#include <sstream>
#endif

#include <visionaray/math/math.h>

#include "macros.h"
#include "sched_common.h"
#include "semaphore.h"

namespace visionaray
{

namespace detail
{

static const int tile_width  = 16;
static const int tile_height = 16;

inline int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

struct sync_params
{
    sync_params()
        : render_loop_active(false)
        , render_loop_exit(false)
    {
    }

    std::mutex mutex;
    std::condition_variable start_render;
    visionaray::semaphore   image_ready;

    std::atomic<long>       tile_idx_counter;
    std::atomic<long>       tile_fin_counter;
    std::atomic<long>       tile_num;

    std::atomic<bool>       render_loop_active;
    std::atomic<bool>       render_loop_exit;
};


} // detail


template <typename R>
struct tiled_sched<R>::impl
{
    typedef std::function<void(recti const&)> render_tile_func;

    impl() = default;

    std::vector<std::thread>    threads;
    detail::sync_params         sync_params;

    recti                       viewport;

    render_tile_func            render_tile;
};

template <typename R>
tiled_sched<R>::tiled_sched()
    : impl_(new impl())
{
    for (unsigned i = 0; i < 8; ++i)
    {
        impl_->threads.push_back( std::thread(&tiled_sched::render_loop, this) );
    }
}

template <typename R>
tiled_sched<R>::~tiled_sched()
{
    impl_->sync_params.render_loop_exit = true;

    for (auto& t : impl_->threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
}

template <typename R>
template <typename K, typename SP>
void tiled_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    impl_->viewport = sched_params.cam.get_viewport();

    typedef typename R::scalar_type     scalar_type;
    typedef matrix<4, 4, scalar_type>   matrix_type;
    typedef typename SP::color_traits   color_traits;
    typedef typename color_traits::type color_type;

    auto inv_view_matrix = matrix_type( inverse(sched_params.cam.get_view_matrix()) );
    auto inv_proj_matrix = matrix_type( inverse(sched_params.cam.get_proj_matrix()) );

    impl_->render_tile = [=](recti const& tile)
    {
        using namespace detail;

        unsigned numx = tile_width  / inc<scalar_type>::x;
        unsigned numy = tile_height / inc<scalar_type>::y;
        for (unsigned i = 0; i < numx * numy; ++i)
        {
            auto pos = vec2ui(i % numx, i / numx);
            auto x = tile.x + pos.x * inc<scalar_type>::x;
            auto y = tile.y + pos.y * inc<scalar_type>::y;

            sample_pixel<R, color_traits>
            (
                x, y, frame_num, inv_view_matrix, inv_proj_matrix,
                sched_params.cam.get_viewport(),
                static_cast<color_type*>(sched_params.rt.color()),
                kernel, typename SP::pixel_sampler_type()
            );
        }
    };

    auto w = impl_->viewport.w - impl_->viewport.x;
    auto h = impl_->viewport.h - impl_->viewport.y;

    auto numtilesx = detail::div_up(w, detail::tile_width);
    auto numtilesy = detail::div_up(h, detail::tile_height);

    auto& sparams = impl_->sync_params;

    sparams.tile_idx_counter = 0;
    sparams.tile_fin_counter = 0;
    sparams.tile_num = numtilesx * numtilesy;

    // render frame
    sparams.render_loop_active = true;
    sparams.start_render.notify_all();

    sparams.image_ready.wait();

    sched_params.rt.end_frame();
}

template <typename R>
void tiled_sched<R>::render_loop()
{
    for (;;)
    {
        auto&       sparams = impl_->sync_params;

        {
            std::unique_lock<std::mutex> l( sparams.mutex );

            while (!sparams.render_loop_active)
            {
                sparams.start_render.wait(l);
            }
        }

        // the actual render loop
        for (;;)
        {
            auto tile_idx = sparams.tile_idx_counter.fetch_add(1);

            if (tile_idx >= sparams.tile_num)
            {
                break;
            }

            auto w = impl_->viewport.w;
            auto tilew = detail::tile_width;
            auto tileh = detail::tile_height;
            auto numtilesx = detail::div_up( w, tilew);

            recti tile
            (
                impl_->viewport.x + (tile_idx % numtilesx) * tilew,
                impl_->viewport.y + (tile_idx / numtilesx) * tileh,
                tilew,
                tileh
            );

            impl_->render_tile(tile);

            auto num_tiles_fin = sparams.tile_fin_counter.fetch_add(1);

            if (num_tiles_fin >= sparams.tile_num - 1)
            {
                assert(num_tiles_fin == sparams.tile_num - 1);
                sparams.image_ready.notify();
                break;
            }
        }

        if (sparams.render_loop_exit)
        {
#ifndef NDEBUG
            std::stringstream str;
            str << "Bye bye\n";
            std::cerr << str.str() << std::endl;
#endif
            break;
        }

    }
}

} // visionaray


