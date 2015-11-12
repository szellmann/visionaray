// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>
#include <utility>

#include "sched_common.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Private implementation
//

template <typename R>
struct simple_sched<R>::impl
{
    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num, std::true_type /* has matrix */);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num, std::false_type /* has matrix */);

    template <typename K, typename SP, typename V, typename ...Args>
    void sample_pixels(K kernel, SP sched_params, unsigned frame_num, V viewport, Args&&... args);
};


// Implementation using matrices and viewport
template <typename R>
template <typename K, typename SP>
void simple_sched<R>::impl::frame(K kernel, SP sched_params, unsigned frame_num, std::true_type)
{
    typedef typename R::scalar_type     scalar_type;
    typedef matrix<4, 4, scalar_type>   matrix_type;

    auto view_matrix            = matrix_type( sched_params.view_matrix );
    auto proj_matrix            = matrix_type( sched_params.proj_matrix );
    auto inv_view_matrix        = matrix_type( inverse(sched_params.view_matrix) );
    auto inv_proj_matrix        = matrix_type( inverse(sched_params.proj_matrix) );
    auto viewport               = sched_params.viewport;

    // Iterate over all pixels
    sample_pixels(
            kernel,
            sched_params,
            frame_num,
            viewport,
            view_matrix,
            inv_view_matrix,
            proj_matrix,
            inv_proj_matrix
            );
}


// Implementation using a pinhole camera
template <typename R>
template <typename K, typename SP>
void simple_sched<R>::impl::frame(K kernel, SP sched_params, unsigned frame_num, std::false_type)
{
    typedef typename R::scalar_type scalar_type;

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sched_params.cam.eye() - sched_params.cam.center() );
    auto s = normalize( cross(sched_params.cam.up(), f) );
    auto u =            cross(f, s);

    auto viewport = sched_params.cam.get_viewport();

    auto eye   = vector<3, scalar_type>(sched_params.cam.eye());
    auto cam_u = vector<3, scalar_type>(s) * scalar_type( tan(sched_params.cam.fovy() / 2.0f) * sched_params.cam.aspect() );
    auto cam_v = vector<3, scalar_type>(u) * scalar_type( tan(sched_params.cam.fovy() / 2.0f) );
    auto cam_w = vector<3, scalar_type>(-f);

    // Iterate over all pixels
    sample_pixels(
            kernel,
            sched_params,
            frame_num,
            viewport,
            eye,
            cam_u,
            cam_v,
            cam_w
            );
}


// Iterate over all pixels in a loop
template <typename R>
template <typename K, typename SP, typename V, typename ...Args>
void simple_sched<R>::impl::sample_pixels(K kernel, SP sched_params, unsigned frame_num, V viewport, Args&&... args)
{
    // TODO: support any sampler
    sampler<typename R::scalar_type> samp(detail::tic());

    for (int y = 0; y < viewport.h; ++y)
    {
        for (int x = 0; x < viewport.w; ++x)
        {
            auto r = detail::make_primary_ray<R>(
                    typename SP::pixel_sampler_type(),
                    samp,
                    x,
                    y,
                    viewport,
                    std::forward<Args>(args)...
                    );

            sample_pixel(
                    kernel,
                    typename SP::pixel_sampler_type(),
                    r,
                    samp,
                    frame_num,
                    sched_params.rt.ref(),
                    x,
                    y,
                    viewport,
                    std::forward<Args>(args)...
                    );
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Simple sched
//

template <typename R>
simple_sched<R>::simple_sched()
    : impl_(new impl)
{
}

template <typename R>
template <typename K, typename SP>
void simple_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    impl_->frame(
            kernel,
            sched_params,
            frame_num,
            typename detail::sched_params_has_view_matrix<SP>::type()
            );
}

} // visionaray
