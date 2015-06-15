// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cuda_runtime_api.h>

#include "sched_common.h"

namespace visionaray
{

namespace detail
{

//-------------------------------------------------------------------------------------------------
// CUDA kernels
//

template <typename R, typename PxSamplerT, typename V, typename RTRef, typename K>
__global__ void render(
        matrix<4, 4, float> view_matrix,
        matrix<4, 4, float> inv_view_matrix,
        matrix<4, 4, float> proj_matrix,
        matrix<4, 4, float> inv_proj_matrix,
        V                   viewport,
        RTRef               rt_ref,
        K                   kernel,
        unsigned            frame_num
        )
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R>(
            kernel,
            PxSamplerT(),
            rt_ref,
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

template <typename R, typename PxSamplerT, typename Vec3, typename V, typename RTRef, typename K>
__global__ void render(
        Vec3        eye,
        Vec3        u,
        Vec3        v,
        Vec3        w,
        V           viewport,
        RTRef       rt_ref,
        K           kernel,
        unsigned    frame_num
        )
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R>(
            kernel,
            PxSamplerT(),
            rt_ref,
            x,
            y,
            frame_num,
            viewport,
            eye,
            u,
            v,
            w
            );
}


//-------------------------------------------------------------------------------------------------
// Dispatch functions
//

template <typename R, typename K, typename SP>
inline void cuda_sched_impl_frame(
        K               kernel,
        SP              sparams,
        unsigned        frame_num,
        std::true_type  /* has matrix */
        )
{
    auto rt_ref             = sparams.rt.ref();
    auto inv_view_matrix    = inverse(sparams.view_matrix);
    auto inv_proj_matrix    = inverse(sparams.proj_matrix);
    auto viewport           = sparams.viewport;

    dim3 block_size(16, 16);

    using cuda_dim_t = decltype(block_size.x);

    auto w = static_cast<cuda_dim_t>(viewport.w);
    auto h = static_cast<cuda_dim_t>(viewport.h);

    dim3 grid_size
    (
        div_up(w, block_size.x),
        div_up(h, block_size.y)
    );
    render<R, typename SP::pixel_sampler_type><<<grid_size, block_size>>>(
            sparams.view_matrix,
            inv_view_matrix,
            sparams.proj_matrix,
            inv_proj_matrix,
            viewport,
            rt_ref,
            kernel,
            frame_num
            );

    cudaPeekAtLastError();
    cudaDeviceSynchronize();
}

template <typename R, typename K, typename SP>
inline void cuda_sched_impl_frame(
        K               kernel,
        SP              sparams,
        unsigned        frame_num,
        std::false_type /* has matrix */
        )
{
    auto rt_ref             = sparams.rt.ref();
    auto viewport           = sparams.cam.get_viewport();

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sparams.cam.eye() - sparams.cam.center() );
    auto s = normalize( cross(sparams.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = sparams.cam.eye();
    auto cam_u = s * tan(sparams.cam.fovy() / 2.0f) * sparams.cam.aspect();
    auto cam_v = u * tan(sparams.cam.fovy() / 2.0f);
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
    render<R, typename SP::pixel_sampler_type><<<grid_size, block_size>>>(
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
}

} // detail


template <typename R>
template <typename K, typename SP>
void cuda_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    detail::cuda_sched_impl_frame<R>(
                kernel,
                sched_params,
                frame_num,
                typename detail::sched_params_has_view_matrix<SP>::type()
                );

    sched_params.rt.end_frame();
}

} // visionaray
