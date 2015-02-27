// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>

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
    typedef std::function<void(recti const&)> render_tile_func;

    impl() = default;

    template <typename K, typename SP>
    void init_render_func(K kernel, SP sched_params, unsigned frame_num);

    template <typename K, typename RT, typename PxSamplerT>
    void init_render_func(K kernel, sched_params<RT, PxSamplerT> sched_params, unsigned frame_num);

    std::vector<std::thread>    threads;
    detail::sync_params         sync_params;

    recti                       viewport;

    render_tile_func            render_tile;
};

template <typename R>
template <typename K, typename SP>
void tiled_sched<R>::impl::init_render_func(K kernel, SP sparams, unsigned frame_num)
{
    // assume that SP has members view_matrix and proj_matrix

    using scalar_type   = typename R::scalar_type;
    using matrix_type   = matrix<4, 4, scalar_type>;

    viewport = sparams.viewport;

    auto inv_view_matrix = matrix_type( inverse(sparams.view_matrix) );
    auto inv_proj_matrix = matrix_type( inverse(sparams.proj_matrix) );

    recti xviewport(viewport.x, viewport.y, viewport.w - 1, viewport.h - 1);

    render_tile = [=](recti const& tile)
    {
        using namespace detail;

        unsigned numx = tile_width  / packet_size<scalar_type>::w;
        unsigned numy = tile_height / packet_size<scalar_type>::h;
        for (unsigned i = 0; i < numx * numy; ++i)
        {
            auto pos = vec2i(i % numx, i / numx);
            auto x = tile.x + pos.x * packet_size<scalar_type>::w;
            auto y = tile.y + pos.y * packet_size<scalar_type>::h;

            recti xpixel(x, y, packet_size<scalar_type>::w - 1, packet_size<scalar_type>::h - 1);
            if ( !overlapping(xviewport, xpixel) )
            {
                continue;
            }

            sample_pixel<R>
            (
                x, y, frame_num, viewport, sparams.rt.color(),
                kernel, typename SP::pixel_sampler_type(),
                inv_view_matrix, inv_proj_matrix
            );
        }
    };
}

template <typename R>
template <typename K, typename RT, typename PxSamplerT>
void tiled_sched<R>::impl::init_render_func(K kernel, sched_params<RT, PxSamplerT> sparams, unsigned frame_num)
{
    // overload for pinhole cam

    using SP            = sched_params<RT, PxSamplerT>;
    using scalar_type   = typename R::scalar_type;
    using color_traits  = typename SP::color_traits;

    viewport = sparams.cam.get_viewport();

    recti xviewport(viewport.x, viewport.y, viewport.w - 1, viewport.h - 1);

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sparams.cam.eye() - sparams.cam.center() );
    auto s = normalize( cross(sparams.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = vector<3, scalar_type>(sparams.cam.eye());
    auto cam_u = vector<3, scalar_type>(s) * scalar_type( tan(sparams.cam.fovy() / 2.0f) * sparams.cam.aspect() );
    auto cam_v = vector<3, scalar_type>(u) * scalar_type( tan(sparams.cam.fovy() / 2.0f) );
    auto cam_w = vector<3, scalar_type>(-f);

    render_tile = [=](recti const& tile)
    {
        using namespace detail;

        unsigned numx = tile_width  / packet_size<scalar_type>::w;
        unsigned numy = tile_height / packet_size<scalar_type>::h;
        for (unsigned i = 0; i < numx * numy; ++i)
        {
            auto pos = vec2i(i % numx, i / numx);
            auto x = tile.x + pos.x * packet_size<scalar_type>::w;
            auto y = tile.y + pos.y * packet_size<scalar_type>::h;

            recti xpixel(x, y, packet_size<scalar_type>::w - 1, packet_size<scalar_type>::h - 1);
            if ( !overlapping(xviewport, xpixel) )
            {
                continue;
            }

            sample_pixel<R>
            (
                x, y, frame_num, viewport, sparams.rt.color(),
                kernel, typename SP::pixel_sampler_type(),
                eye, cam_u, cam_v, cam_w
            );
        }
    };
}



//-------------------------------------------------------------------------------------------------
// tiled_sched implementation
//

template <typename R>
tiled_sched<R>::tiled_sched()
    : impl_(new impl())
{
    for (unsigned i = 0; i < 8; ++i)
    {
        impl_->threads.emplace_back([this](){ render_loop(); });
    }
}

template <typename R>
tiled_sched<R>::~tiled_sched()
{
    impl_->sync_params.render_loop_exit = true;
    impl_->sync_params.threads_start.notify_all();

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

    impl_->init_render_func(kernel, sched_params, frame_num);

    auto w = impl_->viewport.w - impl_->viewport.x;
    auto h = impl_->viewport.h - impl_->viewport.y;

    auto numtilesx = div_up(w, detail::tile_width);
    auto numtilesy = div_up(h, detail::tile_height);

    auto& sparams = impl_->sync_params;

    sparams.tile_idx_counter = 0;
    sparams.tile_fin_counter = 0;
    sparams.tile_num = numtilesx * numtilesy;

    // render frame
    sparams.threads_start.notify_all();

    sparams.threads_ready.wait();

    sched_params.rt.end_frame();
}

template <typename R>
void tiled_sched<R>::render_loop()
{
    for (;;)
    {
        auto& sparams = impl_->sync_params;

        {
            std::unique_lock<std::mutex> l( sparams.mutex );
            sparams.threads_start.wait(l);
        }

    // case event.exit:
        if (sparams.render_loop_exit)
        {
            break;
        }

    // case event.render:
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
            auto numtilesx = div_up( w, tilew );

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
                sparams.threads_ready.notify();
                break;
            }
        }
    }
}

} // visionaray
