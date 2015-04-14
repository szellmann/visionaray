// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "sched_common.h" // TODO: consolidate

namespace visionaray
{

template <typename R>
template <typename K, typename SP>
void simple_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    typedef typename R::scalar_type     scalar_type;
    typedef matrix<4, 4, scalar_type>   matrix_type;

    sched_params.rt.begin_frame();

    auto view_matrix            = matrix_type( sched_params.view_matrix );
    auto proj_matrix            = matrix_type( sched_params.proj_matrix );
    auto inv_view_matrix        = matrix_type( inverse(sched_params.view_matrix) );
    auto inv_proj_matrix        = matrix_type( inverse(sched_params.proj_matrix) );
    auto viewport               = sched_params.viewport;

    for (int y = 0; y < viewport.h; ++y)
    {
        for (int x = 0; x < viewport.w; ++x)
        {
            sample_pixel<R>(
                    x,
                    y,
                    frame_num,
                    viewport,
                    sched_params.rt.ref(),
                    kernel,
                    typename SP::pixel_sampler_type(),
                    view_matrix,
                    inv_view_matrix,
                    proj_matrix,
                    inv_proj_matrix
                    );
        }
    }

    sched_params.rt.end_frame();
}

} // visionaray
