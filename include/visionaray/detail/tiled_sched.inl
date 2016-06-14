// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <atomic>
#include <cassert>
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
    // TODO: any sampler
    typedef std::function<void(recti const&, sampler<typename R::scalar_type>&)> render_tile_func;

    void init_threads(unsigned num_threads);
    void destroy_threads();

    void render_loop();

    template <typename K, typename SP>
    void init_render_func(K kernel, SP sparams, unsigned frame_num, std::true_type /* has matrix */);

    template <typename K, typename SP>
    void init_render_func(K kernel, SP sparams, unsigned frame_num, std::false_type /* has matrix */);

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
                args...
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
                args...
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

    recti                       viewport;
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
        sampler<typename R::scalar_type> samp(detail::tic());
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
void tiled_sched<R>::impl::init_render_func(K kernel, SP sparams, unsigned frame_num, std::true_type)
{
    // overload for two matrices and a viewport

    using T = typename R::scalar_type;
    using matrix_type   = matrix<4, 4, T>;

    viewport    = sparams.viewport;
    scissor_box = sparams.scissor_box;

    auto view_matrix     = matrix_type( sparams.view_matrix );
    auto proj_matrix     = matrix_type( sparams.proj_matrix );
    auto inv_view_matrix = matrix_type( inverse(sparams.view_matrix) );
    auto inv_proj_matrix = matrix_type( inverse(sparams.proj_matrix) );

    recti clip_rect(scissor_box.x, scissor_box.y, scissor_box.w - 1, scissor_box.h - 1);

    render_tile = [=](recti const& tile, sampler<T>& samp)
    {
        using namespace detail;

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
                    viewport.w,
                    viewport.h,
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

    using T = typename R::scalar_type;

    viewport    = sparams.cam.get_viewport();
    scissor_box = sparams.scissor_box;

    recti clip_rect(scissor_box.x, scissor_box.y, scissor_box.w - 1, scissor_box.h - 1);

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sparams.cam.eye() - sparams.cam.center() );
    auto s = normalize( cross(sparams.cam.up(), f) );
    auto u =            cross(f, s);

    vec3 eye   = sparams.cam.eye();
    vec3 cam_u = s * tan(sparams.cam.fovy() / 2.0f) * sparams.cam.aspect();
    vec3 cam_v = u * tan(sparams.cam.fovy() / 2.0f);
    vec3 cam_w = -f;

    render_tile = [=](recti const& tile, sampler<T>& samp)
    {
        using namespace detail;

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
                    viewport.w,
                    viewport.h,
                    vector<3, T>(eye),
                    vector<3, T>(cam_u),
                    vector<3, T>(cam_v),
                    vector<3, T>(cam_w)
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

    auto w = impl_->viewport.w;
    auto h = impl_->viewport.h;

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
