// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "tiled_sched.h" // TODO: consolidate

namespace visionaray
{

template <typename R>
template <typename K, typename SP>
void simple_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    typedef typename SP::color_type     color_type;
    typedef typename R::scalar_type     scalar_type;
    typedef matrix<4, 4, scalar_type>   matrix_type;

    sched_params.rt->begin_frame();

    color_type* color_buffer    = static_cast<color_type*>(sched_params.rt->color());
    auto inv_view_matrix        = matrix_type( inverse(sched_params.cam.get_view_matrix()) );
    auto inv_proj_matrix        = matrix_type( inverse(sched_params.cam.get_proj_matrix()) );
    auto viewport               = sched_params.cam.get_viewport();

    for (int y = 0; y < viewport.h; ++y)
    {
        for (int x = 0; x < viewport.w; ++x)
        {
            detail::sample_pixel<R>(x, y, frame_num, inv_view_matrix, inv_proj_matrix,
                viewport, color_buffer, kernel, typename SP::pixel_sampler_type());
        }
    }

    sched_params.rt->end_frame();
}

} // visionaray


