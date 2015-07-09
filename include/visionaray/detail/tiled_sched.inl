// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <utility>

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

    void init_threads(unsigned num_threads);
    void destroy_threads();

    void render_loop();

    template <typename K, typename SP>
    void init_render_func(K kernel, SP sparams, unsigned frame_num, std::true_type /* has matrix */);

    template <typename K, typename SP>
    void init_render_func(K kernel, SP sparams, unsigned frame_num, std::false_type /* has matrix */);

    template <typename K, typename SP, typename ...Args>
    void call_sample_pixel(
            std::false_type /* has intersector */,
            K               kernel,
            SP              sparams,
            Args&&...       args
            )
    {
        sample_pixel<R>(
                kernel,
                typename SP::pixel_sampler_type(),
                sparams.rt.ref(),
                std::forward<Args>(args)...
                );
    }

    template <typename K, typename SP, typename ...Args>
    void call_sample_pixel(
            std::true_type  /* has intersector */,
            K               kernel,
            SP              sparams,
            Args&&...       args)
    {
        sample_pixel<R>(
                detail::have_intersector_tag(),
                sparams.intersector,
                kernel,
                typename SP::pixel_sampler_type(),
                sparams.rt.ref(),
                std::forward<Args>(args)...
                );
    }

    std::vector<std::thread>    threads;
    detail::sync_params         sync_params;

    recti                       viewport;

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
        for (;;)
        {
            auto tile_idx = sync_params.tile_idx_counter.fetch_add(1);

            if (tile_idx >= sync_params.tile_num)
            {
                break;
            }

            auto w = viewport.w;
            auto tilew = detail::tile_width;
            auto tileh = detail::tile_height;
            auto numtilesx = div_up( w, tilew );

            recti tile
            (
                viewport.x + (tile_idx % numtilesx) * tilew,
                viewport.y + (tile_idx / numtilesx) * tileh,
                tilew,
                tileh
            );

            render_tile(tile);

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
void tiled_sched<R>::impl::init_render_func(K kernel, SP sparams, unsigned frame_num, std::true_type)
{
    using scalar_type   = typename R::scalar_type;
    using matrix_type   = matrix<4, 4, scalar_type>;

    viewport = sparams.viewport;

    auto view_matrix     = matrix_type( sparams.view_matrix );
    auto proj_matrix     = matrix_type( sparams.proj_matrix );
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

            call_sample_pixel(
                    typename detail::sched_params_has_intersector<SP>::type(),
                    kernel,
                    sparams,
                    x,
                    y,
                    frame_num,
                    viewport,
                    view_matrix,
                    inv_view_matrix,
                    proj_matrix,
                    inv_proj_matrix
                    );
        }
    };
}

template <typename R>
template <typename K, typename SP>
void tiled_sched<R>::impl::init_render_func(K kernel, SP sparams, unsigned frame_num, std::false_type)
{
    // overload for pinhole cam

    using scalar_type   = typename R::scalar_type;

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

            call_sample_pixel(
                    typename detail::sched_params_has_intersector<SP>::type(),
                    kernel,
                    sparams,
                    x,
                    y,
                    frame_num,
                    viewport,
                    eye,
                    cam_u,
                    cam_v,
                    cam_w
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
    sched_params.rt.begin_frame();

    impl_->init_render_func(
            kernel,
            sched_params,
            frame_num,
            typename detail::sched_params_has_view_matrix<SP>::type()
            );

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
void tiled_sched<R>::set_num_threads(unsigned num_threads)
{
    if ( get_num_threads()  == num_threads )
    {
        return;
    }

    impl_->destroy_threads();
    impl_->init_threads(num_threads);
}

template <typename R>
unsigned tiled_sched<R>::get_num_threads() const
{
    return static_cast<unsigned>( impl_->threads.size() );
}

} // visionaray
