// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cuda_runtime_api.h>

#include "tiled_sched.h" // TODO: consolidate

namespace visionaray
{

namespace detail
{

template <typename R, typename CT, typename PxSamplerT, typename MT, typename V, typename C, typename K>
__global__ void render(MT inv_view_matrix, MT inv_proj_matrix, V viewport,
    C* color_buffer, K kernel, unsigned frame)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R, CT>(x, y, frame, viewport, color_buffer, kernel, PxSamplerT(), inv_view_matrix, inv_proj_matrix);
}

} // detail



template <typename R>
template <typename K, typename SP>
void cuda_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    typedef typename SP::color_traits   color_traits;
    typedef typename color_traits::type color_type;
    auto color_buffer       = sched_params.rt.color();
    auto inv_view_matrix    = inverse(sched_params.cam.get_view_matrix());
    auto inv_proj_matrix    = inverse(sched_params.cam.get_proj_matrix());
    auto viewport           = sched_params.cam.get_viewport();

    dim3 block_size(16, 16);
    dim3 grid_size
    (
        detail::div_up(viewport.w, block_size.x),
        detail::div_up(viewport.h, block_size.y)
    );
    detail::render<R, color_traits, typename SP::pixel_sampler_type><<<grid_size, block_size>>>
    (
        inv_view_matrix,
        inv_proj_matrix,
        viewport,
        color_buffer,
        kernel,
        frame_num
    );

    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    sched_params.rt.end_frame();
}

} // visionaray


