// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>
#include <utility>

#include <visionaray/random_sampler.h>

#include "sched_common.h"

namespace visionaray
{

namespace detail
{
namespace simple
{

//-------------------------------------------------------------------------------------------------
// Private functions
//

// Iterate over all pixels in a loop
template <typename R, typename K, typename SP, typename ...Args>
void sample_pixels_impl(R /* */, K kernel, SP sched_params, unsigned frame_num, Args&&... args)
{
    // TODO: support any sampler
    random_sampler<typename R::scalar_type> samp(detail::tic());

    for (size_t y = 0; y < sched_params.rt.height(); ++y)
    {
        for (size_t x = 0; x < sched_params.rt.width(); ++x)
        {
            auto r = detail::make_primary_rays(
                    R{},
                    typename SP::pixel_sampler_type{},
                    samp,
                    x,
                    y,
                    std::forward<Args>(args)...
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
                    std::forward<Args>(args)...
                    );
        }
    }
}


// Implementation using matrices
template <typename R, typename K, typename SP>
void frame_impl(R /* */, K kernel, SP sched_params, unsigned frame_num, std::true_type)
{
    typedef typename R::scalar_type     scalar_type;
    typedef matrix<4, 4, scalar_type>   matrix_type;

    matrix_type view_matrix( sched_params.view_matrix );
    matrix_type proj_matrix( sched_params.proj_matrix );
    matrix_type inv_view_matrix( inverse(sched_params.view_matrix) );
    matrix_type inv_proj_matrix( inverse(sched_params.proj_matrix) );

    // Iterate over all pixels
    sample_pixels_impl(
            R{},
            kernel,
            sched_params,
            frame_num,
            sched_params.rt.width(),
            sched_params.rt.height(),
            view_matrix,
            inv_view_matrix,
            proj_matrix,
            inv_proj_matrix
            );
}


// Implementation using a pinhole camera
template <typename R, typename K, typename SP>
void frame_impl(R /* */, K kernel, SP sched_params, unsigned frame_num, std::false_type)
{
    typedef typename R::scalar_type scalar_type;

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sched_params.cam.eye() - sched_params.cam.center() );
    auto s = normalize( cross(sched_params.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = vector<3, scalar_type>(sched_params.cam.eye());
    auto cam_u = vector<3, scalar_type>(s) * scalar_type( tan(sched_params.cam.fovy() / 2.0f) * sched_params.cam.aspect() );
    auto cam_v = vector<3, scalar_type>(u) * scalar_type( tan(sched_params.cam.fovy() / 2.0f) );
    auto cam_w = vector<3, scalar_type>(-f);

    // Iterate over all pixels
    sample_pixels_impl(
            R{},
            kernel,
            sched_params,
            frame_num,
            sched_params.rt.width(),
            sched_params.rt.height(),
            eye,
            cam_u,
            cam_v,
            cam_w
            );
}

} // simple
} // detail


//-------------------------------------------------------------------------------------------------
// Simple sched
//

template <typename R>
template <typename K, typename SP>
void simple_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    detail::simple::frame_impl(
            R{},
            kernel,
            sched_params,
            frame_num,
            typename detail::sched_params_has_view_matrix<SP>::type{}
            );
}

} // visionaray
