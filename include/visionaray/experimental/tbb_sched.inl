// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

namespace visionaray
{

namespace detail
{

template <class F>
void tile2d(int x0, int x1, int y0, int y1, int dx, int dy, F f)
{
    //assert((x1 - x0) % tile_w == 0);
    //assert((y1 - y0) % tile_h == 0);

    for (int j = y0; j < y1; j += dy)
    {
        for (int i = x0; i < x1; i += dx)
        {
            f(i, i + dx, j, j + dy);
        }
    }
}

} // detail

template <typename R>
struct tbb_sched<R>::impl
{
    tbb::task_scheduler_init init_;

    impl(int num_threads)
        : init_(num_threads)
    {
    }
};

template <typename R>
tbb_sched<R>::tbb_sched(int num_threads)
    : impl_(new impl(num_threads))
{
}

template <typename R>
tbb_sched<R>::~tbb_sched()
{
}

template <typename R>
template <typename K, typename SP>
void tbb_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    using scalar_type   = typename R::scalar_type;
    using color_traits  = typename SP::color_traits;
    using matrix_type   = matrix<4, 4, scalar_type>;

    auto viewport = sched_params.cam.get_viewport();

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sched_params.cam.eye() - sched_params.cam.center() );
    auto s = normalize( cross(sched_params.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = vector<3, scalar_type>(sched_params.cam.eye());
    auto cam_u = vector<3, scalar_type>(s) * scalar_type( tan(sched_params.cam.fovy() / 2.0f) * sched_params.cam.aspect() );
    auto cam_v = vector<3, scalar_type>(u) * scalar_type( tan(sched_params.cam.fovy() / 2.0f) );
    auto cam_w = vector<3, scalar_type>(-f);

    int pw = detail::packet_size<scalar_type>::w;
    int ph = detail::packet_size<scalar_type>::h;

    // Tile size must be be a multiple of packet size.
#if 1
    int dx = round_up(16, pw);
    int dy = round_up(16, ph);
#else
    int dx = 8 * pw;
    int dy = 8 * ph;
#endif

    int nx = round_up(viewport.w, dx);
    int ny = round_up(viewport.h, dy);

    sched_params.rt.begin_frame();

    tbb::parallel_for( tbb::blocked_range2d<int>(0, nx, dx, 0, ny, dy),
        [=](tbb::blocked_range2d<int> const& r)
        {
            using namespace detail;

            int x0 = r.rows().begin();
            int x1 = r.rows().end();
            int y0 = r.cols().begin();
            int y1 = r.cols().end();

            tile2d( x0, x1, y0, y1, pw, ph,
                [=](int i0, int /*i1*/, int j0, int /*j1*/)
                {
                    sample_pixel<R>(
                        i0,
                        j0,
                        frame_num,
                        viewport,
                        sched_params.rt.ref(),
                        kernel,
                        typename SP::pixel_sampler_type(),
                        eye,
                        cam_u,
                        cam_v,
                        cam_w
                        );
                }
            );
        }
    );

    sched_params.rt.end_frame();
}

} // visionaray
