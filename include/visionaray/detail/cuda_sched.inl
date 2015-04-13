// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cuda_runtime_api.h>

#include "sched_common.h"

namespace visionaray
{

namespace detail
{

template <typename R, typename PxSamplerT, typename MT, typename V, typename RTRef, typename K>
__global__ void render(
        MT          inv_view_matrix,
        MT          inv_proj_matrix,
        V           viewport,
        RTRef       rt_ref,
        K           kernel,
        unsigned    frame
        )
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R>(
            x,
            y,
            frame,
            viewport,
            rt_ref,
            kernel,
            PxSamplerT(),
            inv_view_matrix,
            inv_proj_matrix
            );
}

template <typename R, typename PxSamplerT, typename Vec3, typename V, typename RTRef, typename K>
__global__ void render(
        Vec3        eye,
        Vec3        u,
        Vec3        v,
        Vec3        w,
        V           viewport,
        RTRef       rt_ref,
        K           kernel,
        unsigned    frame
        )
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R>(
            x,
            y,
            frame,
            viewport,
            rt_ref,
            kernel,
            PxSamplerT(),
            eye,
            u,
            v,
            w
            );
}

} // detail



template <typename R>
template <typename K, typename SP>
void cuda_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    auto rt_ref             = sched_params.rt.ref();
//  auto inv_view_matrix    = inverse(sched_params.cam.get_view_matrix());
//  auto inv_proj_matrix    = inverse(sched_params.cam.get_proj_matrix());
    auto viewport           = sched_params.cam.get_viewport();

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sched_params.cam.eye() - sched_params.cam.center() );
    auto s = normalize( cross(sched_params.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = sched_params.cam.eye();
    auto cam_u = s * tan(sched_params.cam.fovy() / 2.0f) * sched_params.cam.aspect();
    auto cam_v = u * tan(sched_params.cam.fovy() / 2.0f);
    auto cam_w = -f;

    dim3 block_size(16, 16);

    using cuda_dim_t = decltype(block_size.x);

    auto w = static_cast<cuda_dim_t>(viewport.w);
    auto h = static_cast<cuda_dim_t>(viewport.h);

    dim3 grid_size
    (
        div_up(w, block_size.x),
        div_up(h, block_size.y)
    );
    detail::render<R, typename SP::pixel_sampler_type><<<grid_size, block_size>>>(
//          inv_view_matrix,
//          inv_proj_matrix,
            eye,
            cam_u,
            cam_v,
            cam_w,
            viewport,
            rt_ref,
            kernel,
            frame_num
            );

    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    sched_params.rt.end_frame();
}

} // visionaray
